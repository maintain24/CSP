import os
import time
import random
import numpy as np
import pandas as pd
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config
from util.s3dis import S3DIS
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name == 's3dis':
        S3DIS(split='train', data_root=args.data_root, test_area=args.test_area)
        S3DIS(split='val', data_root=args.data_root, test_area=args.test_area)
    else:
        raise NotImplementedError()
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    '''
    导入模型model
    '''
    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    elif args.arch == 'DNN':  # 模型替换
        from model.pointtransformer.DNN import pointtransformer_seg_repro as Model
    elif args.arch == 'Unet':  # 模型替换
        print("OK!!!!!!!!!")
        from model.pointtransformer.Unet import unet_pointtransformer_seg as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes)
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)

    '''
    这段代码实现了分布式训练的功能。如果主进程（main_process）返回True，则获取日志记录器（logger）和摘要写入器（writer），打印出参数、类别和模型信息。
    如果使用分布式训练，则设置当前GPU设备，调整批量大小、验证批量大小和工作线程数，并使用torch.nn.parallel.DistributedDataParallel函数将模型并行化处理。
    '''
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        print('the using gpus are：', gpu)
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )

    else:
        model = torch.nn.DataParallel(model.cuda())

    '''
    这段代码实现了从指定路径加载预训练权重（weight）的功能。如果参数args.weight存在，则判断该路径是否为文件（os.path.isfile），
    如果是文件则在主进程中记录日志信息，加载权重（model.load_state_dict）并在主进程中记录日志信息，否则在主进程中记录日志信息表示未找到权重文件。
    '''
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    '''
    这段代码实现了从指定路径加载训练过程中的检查点（checkpoint）的功能。如果参数args.resume存在，则判断该路径是否为文件（os.path.isfile），
    如果是文件则在主进程中记录日志信息，加载检查点（torch.load），加载检查点中的epoch、模型状态、优化器状态、学习率调度器状态和最佳IOU，
    并在主进程中记录日志信息表示已加载检查点，否则在主进程中记录日志信息表示未找到检查点文件。
    '''
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            #best_iou = 40.0
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    '''
    这段代码实现了数据增强的功能，具体包括随机缩放（RandomScale）、自动对比度调整（ChromaticAutoContrast）、
    颜色平移（ChromaticTranslation）、颜色抖动（ChromaticJitter）和色调饱和度平移（HueSaturationTranslation）。
    定义Dataset和DataLoader
    '''
    train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)

    val_loader = None
    if args.evaluate:
        val_transform = None
        val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)

    # filename = r'/mnt/Dataset/PT2/result/result.txt'  # 写入txt，并将下面的for循环缩进
    # with open(filename, 'a+') as f:  # 写入txt
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
        # f.write(str(loss_train) + " ")      # 写入txt
        # f.write(str(mIoU_train) + " ")      # 写入txt
        # f.write(str(mAcc_train) + " ")      # 写入txt
        # f.write(str(allAcc_train) + "\n")   # 写入txt
        scheduler.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'shapenet':
                raise NotImplementedError()
            else:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))

    # f.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    filename = r'/mnt/Dataset/PT2_1/result/result.txt'
    xyz_file = r'/mnt/Dataset/PT2_1/result/coordinate.csv'  # 保存预测坐标数据集
    x_file_dir = r'/mnt/Dataset/PT2_1/result'  # 保存预测坐标数据集
    """保存中间变量的空ndarray"""
    p0_all = np.random.rand(0, 3)  # 创建空ndarray
    p2_all = np.random.rand(0, 3)
    p3_all = np.random.rand(0, 3)
    p4_all = np.random.rand(0, 3)
    p5_all = np.random.rand(0, 3)
    o0_all = np.random.rand(0)  # 保存标签
    o2_all = np.random.rand(0)
    o3_all = np.random.rand(0)
    o4_all = np.random.rand(0)
    o5_all = np.random.rand(0)
    x1_all = np.random.rand(0, 8)  # 创建空ndarray
    x2_all = np.random.rand(0, 8)
    x3_all = np.random.rand(0, 8)
    x4_all = np.random.rand(0, 8)
    x1_new_all = np.random.rand(0, 8)
    x2_new_all = np.random.rand(0, 8)
    x3_new_all = np.random.rand(0, 8)
    x4_new_all = np.random.rand(0, 8)


    with open(filename, 'a+') as f:
        for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
            data_time.update(time.time() - end)
            coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)

            """
            输出变量增加，得到新output
            """
            # 试图输出更多的结果，但最后输出发现结果不尽人意
            p0, o0, p1, o1, p2, o2, p3, o3, p4, o4, p5, o5, x, x1, x1_new, x2, x2_new, x3, x3_new, x4, x4_new\
                = model([coord, feat, offset])  # (n, 3), (n, c), (b) -> (n, c)  # x = model([coord, feat, offset])
            output = x  # 原output只输出一个元组x（n,c），output = model([coord, feat, offset])

            if i == epoch + 1:
                """处理中间输出变量p和o"""
                p0_np = p0.detach().cpu().numpy()  # 下采样过程坐标变化
                p2_np = p2.detach().cpu().numpy()
                p3_np = p3.detach().cpu().numpy()
                p4_np = p4.detach().cpu().numpy()
                p5_np = p5.detach().cpu().numpy()
                o0_np = o0.detach().cpu().numpy()
                o2_np = o2.detach().cpu().numpy()
                o3_np = o3.detach().cpu().numpy()
                o4_np = o4.detach().cpu().numpy()
                o5_np = o5.detach().cpu().numpy()

                p0_all = np.append(p0_all, p0_np, axis=0)
                p2_all = np.append(p2_all, p2_np, axis=0)
                p3_all = np.append(p3_all, p3_np, axis=0)
                p4_all = np.append(p4_all, p4_np, axis=0)
                p5_all = np.append(p5_all, p5_np, axis=0)
                o0_all = np.append(o0_all, o0_np, axis=0)
                o2_all = np.append(o2_all, o2_np, axis=0)
                o3_all = np.append(o3_all, o3_np, axis=0)
                o4_all = np.append(o4_all, o4_np, axis=0)
                o5_all = np.append(o5_all, o5_np, axis=0)

                """处理中间输出变量x和x_new"""
                x1_np = x1.detach().cpu().numpy()  # 下采样过程特征变化
                x1_new_np = x1_new.detach().cpu().numpy()
                x2_np = x2.detach().cpu().numpy()
                x2_new_np = x2_new.detach().cpu().numpy()
                x3_np = x3.detach().cpu().numpy()
                x3_new_np = x3_new.detach().cpu().numpy()
                x4_np = x4.detach().cpu().numpy()
                x4_new_np = x4_new.detach().cpu().numpy()

                x1_all = np.append(x1_all, x1_np, axis=0)
                x1_new_all = np.append(x1_new_all, x1_new_np, axis=0)
                x2_all = np.append(x2_all, x2_np, axis=0)
                x2_new_all = np.append(x2_new_all, x2_new_np, axis=0)
                x3_all = np.append(x3_all, x3_np, axis=0)
                x3_new_all = np.append(x3_new_all, x3_new_np, axis=0)
                x4_all = np.append(x4_all, x4_np, axis=0)
                x4_new_all = np.append(x4_new_all, x4_new_np, axis=0)

            """原output"""
            # output = model([coord, feat, offset])

            # print('<<<<<<<<<<<<<<<<<<<<<<<<<<调试输出>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            # print('p:', p)
            # print('coord:', coord)
            # print('coord.device:', coord_np.device)  # 输出应该是 "cpu"
            # print('p.device:', p_np.device)  # 输出应该是 "cpu"
            # np.savetxt(xyz_file, np.concatenate((p0_np, p2_np, p3_np, p4_np, p5_np), axis=1),
            #            delimiter=',', fmt='%.6f')  # 保存预测坐标数据集
            # np.savetxt(xyz_file, np.concatenate((coord_np, p0_np, p1_np, p2_np, p3_np, p4_np, p5_np), axis=1),
            #            delimiter=',', fmt='%.6f')  # 保存预测坐标数据集
            if target.shape[-1] == 1:
                target = target[:, 0]  # for cls
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.max(1)[1]
            n = coord.size(0)
            """
            这里的output可以试着保存下来。举例来说，如果当前 batch 的大小为 4，output 的值为 [1, 0, 2, 1]，
            则表示第一个样本的预测类别为 1，第二个样本的预测类别为 0，第三个样本的预测类别为 2，第四个样本的预测类别为 1。
            """
            if args.multiprocessing_distributed:
                loss *= n
                count = target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count)
                n = count.item()
                loss /= n
            intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), n)
            batch_time.update(time.time() - end)
            end = time.time()

            # calculate remain time
            current_iter = epoch * len(train_loader) + i + 1
            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'Loss {loss_meter.val:.4f} '
                            'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                              batch_time=batch_time, data_time=data_time,
                                                              remain_time=remain_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))
            f.write(str(epoch + 1) + " ")       # 写入txt
            f.write(str(loss_meter.val) + " ")  # 写入txt
            f.write(str(accuracy) + "\n")       # 写入txt

            if main_process():
                writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
                writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
                writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
                writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

        """保存过程变量坐标p和偏移量o（又称结束索引）"""
        # 将每个ndarray数据转化为DataFrame格式
        df1 = pd.DataFrame(p0_all)
        df2 = pd.DataFrame(o0_all)
        df3 = pd.DataFrame(p2_all)
        df4 = pd.DataFrame(o2_all)
        df5 = pd.DataFrame(p3_all)
        df6 = pd.DataFrame(o3_all)
        df7 = pd.DataFrame(p4_all)
        df8 = pd.DataFrame(o4_all)
        df9 = pd.DataFrame(p5_all)
        df10 = pd.DataFrame(o5_all)
        # 使用concat函数按列拼接这7个DataFrame
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], axis=1)
        # 将拼接后的DataFrame保存到csv文件中
        df.to_csv(xyz_file, index=False, header=False)

        """保存特征张量x和x_new"""
        # 将每个ndarray数据转化为DataFrame格式
        df11 = pd.DataFrame(x1_all)
        df12 = pd.DataFrame(x1_new_all)
        df13 = pd.DataFrame(x2_all)
        df14 = pd.DataFrame(x2_new_all)
        df15 = pd.DataFrame(x3_all)
        df16 = pd.DataFrame(x3_new_all)
        df17 = pd.DataFrame(x4_all)
        df18 = pd.DataFrame(x4_new_all)
        # 将拼接后的DataFrame保存到csv文件中
        df11.to_csv("{}/{}.csv".format(x_file_dir, 'x1'), index=False, header=False)
        df12.to_csv("{}/{}.csv".format(x_file_dir, 'x1_new'), index=False, header=False)
        df13.to_csv("{}/{}.csv".format(x_file_dir, 'x2'), index=False, header=False)
        df14.to_csv("{}/{}.csv".format(x_file_dir, 'x2_new'), index=False, header=False)
        df15.to_csv("{}/{}.csv".format(x_file_dir, 'x3'), index=False, header=False)
        df16.to_csv("{}/{}.csv".format(x_file_dir, 'x3_new'), index=False, header=False)
        df17.to_csv("{}/{}.csv".format(x_file_dir, 'x4'), index=False, header=False)
        df18.to_csv("{}/{}.csv".format(x_file_dir, 'x4_new'), index=False, header=False)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    f.close()  # 写入txt
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        with torch.no_grad():
            # 试图输出更多结果，但发现不尽人意
            p0, o0, p1, o1, p2, o2, p3, o3, p4, o4, p5, o5, x, x1, x1_new, x2, x2_new, x3, x3_new, x4, x4_new\
                = model([coord, feat, offset])  # output = model([coord, feat, offset]),但扩展为元组无法log_softmax
            output = x

            # output = model([coord, feat, offset])
        loss = criterion(output, target)

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
