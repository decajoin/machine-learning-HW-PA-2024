import pyecharts.options as opts
from pyecharts.charts import Line
import pandas as pd

# 读取保存的损失值CSV文件
loss_df = pd.read_csv('./Data/losses.csv')
test_loss_df = pd.read_csv('./Data/test_losses.csv')

# 创建折线图
line = (
    Line()
    .add_xaxis(loss_df['iteration'].tolist())  # x轴为迭代次数
    .add_yaxis(
        series_name="Train Loss",  # 系列名称
        y_axis=loss_df['loss'].tolist(),  # y轴为损失值
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),  # 区域样式
        label_opts=opts.LabelOpts(is_show=False),  # 不显示标签
    )
    .add_yaxis(
        series_name="Test Loss",
        y_axis=test_loss_df['test_loss'].tolist(),  # y轴为测试损失值
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),  # 区域样式
        label_opts=opts.LabelOpts(is_show=False),  # 不显示标签
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="迭代次数与损失值关系图"),  # 图表标题
        xaxis_opts=opts.AxisOpts(name="迭代次数"),  # x轴名称
        yaxis_opts=opts.AxisOpts(name="损失值"),  # y轴名称
    )
)

# 渲染图表到HTML文件
line.render("./Pics/迭代次数与损失值关系图.html")
