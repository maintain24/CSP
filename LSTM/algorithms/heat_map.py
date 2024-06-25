from matplotlib import font_manager
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class CyrusPlot(object):
    def __init__(self,dpi=72,fig_size=[10,10]):
        """
        实列化该类，然后直接调用cyrus_heat_map方法
        :param dpi:
        :param fig_size:
        """
        self.dpi = dpi
        self.fig_size = fig_size
        self.font = font_manager.FontProperties(fname=r"new.csv", size=30)  # mnt\pycharm_project_3\algorithms\

    def cyrus_heat_map(self,datas,x_ticks = [],y_ticks = [],bar_label = "bar label",show = True,save_name = ""):
        figure = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        ax = figure.add_subplot(111)
        if not x_ticks:
            x_ticks = ["x"+str(i) for i in range(datas.shape[1])]
            y_ticks = ["y" + str(i) for i in range(datas.shape[0])]
        im, _ = self.heatmap(np.array(datas), x_ticks, y_ticks,
                        cmap="RdBu", cbarlabel=bar_label,ax=ax)  # plt.cm.RdBu   PuOr
        self.annotate_heatmap(im, valfmt="{x:.2f}", size=16)
        if save_name:
            plt.savefig("./figure/" + save_name + ".jpg")
        if show:
            plt.show()

    def heatmap(self,data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        if not ax:
            ax = plt.gca()
        im = ax.imshow(data, **kwargs)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontproperties=font_manager.FontProperties(fname="C:\Windows\Fonts\simhei.ttf", size=30))
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels,fontproperties=self.font)
        ax.set_yticklabels(row_labels,fontproperties=self.font)
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(self,im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        kw = dict(horizontalalignment="center",
                  verticalalignment="center",
                  )
        kw.update(textkw)

        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[abs(data[i, j]) > 0.5])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

if __name__ == '__main__':
    # 构造数据集并计算其pearson相关系数
    data = pd.DataFrame(np.random.randn(243,243))
    pearson = data.corr()
    plot_tool = CyrusPlot()
    plot_tool.cyrus_heat_map(pearson,show=True)

