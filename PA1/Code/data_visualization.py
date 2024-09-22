import pandas as pd
from pyecharts.charts import Scatter
from pyecharts import options as opts

# 读取数据
df = pd.read_csv('./Data/advertising.csv')

# 创建散点图
def create_scatter_chart(x_data, y_data, x_label):
    scatter = (
        Scatter()
        .add_xaxis(x_data)
        .add_yaxis("Sales", y_data)
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{x_label} vs Sales"),
            xaxis_opts=opts.AxisOpts(name=x_label),
            yaxis_opts=opts.AxisOpts(name="Sales"),
        )
    )
    return scatter

# 按升序排序每个平台的花费再绘图
wechat_sorted = df.sort_values(by='wechat')
wechat_spend_sorted = wechat_sorted['wechat'].tolist()
wechat_sales_sorted = wechat_sorted['sales'].tolist()

weibo_sorted = df.sort_values(by='weibo')
weibo_spend_sorted = weibo_sorted['weibo'].tolist()
weibo_sales_sorted = weibo_sorted['sales'].tolist()

others_sorted = df.sort_values(by='others')
others_spend_sorted = others_sorted['others'].tolist()
others_sales_sorted = others_sorted['sales'].tolist()


wechat_chart = create_scatter_chart(wechat_spend_sorted, wechat_sales_sorted, "WeChat Spend")
weibo_chart = create_scatter_chart(weibo_spend_sorted, weibo_sales_sorted, "Weibo Spend")
others_chart = create_scatter_chart(others_spend_sorted, others_sales_sorted, "Others Spend")

# Step 5: Render charts to HTML files
wechat_chart.render("./Pics/微信.html")
weibo_chart.render("./Pics/微博.html")
others_chart.render("./Pics/其他.html")
