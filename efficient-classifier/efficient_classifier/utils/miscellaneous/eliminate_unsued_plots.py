def eliminate_unused_plots(fig, axes, i):
            for j in range(i + 1, len(axes)):
                  fig.delaxes(axes[j])