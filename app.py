import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import talib
import numpy as np

# ----------------------------- 页面初始化 -----------------------------
st.set_page_config(page_title="市场脉搏监控系统", layout="wide")
st.title("📈 市场脉搏监控系统")
st.markdown("监控关键指标，评估 QQQ/SPY 短期走势 | 综合评分：0-100 (分数越高越乐观)")
st.markdown("---")

# ----------------------------- 侧边栏控制 -----------------------------
with st.sidebar:
    st.header("控制面板")
    ticker_option = st.selectbox("选择标的", ["QQQ", "SPY"], index=0)
    lookback_days = st.slider("回看天数", min_value=30, max_value=200, value=90)

    st.markdown("---")
    st.markdown("**风险提示**")
    st.info("""
    本工具仅为量化指标分析仪表板。
    **不构成任何投资建议。**
    市场有风险，决策需谨慎。
    """)
    st.markdown("---")
    st.caption(f"数据更新至: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 根据选择确定标的
ticker_symbol = ticker_option
bond_ticker = "HYG"  # 使用 iShares 高收益公司债 ETF 作为垃圾债代理
vix_ticker = "^VIX"  # VIX指数

# ----------------------------- 数据获取函数 -----------------------------
@st.cache_data(ttl=3600)  # 缓存数据1小时，减少API调用
def fetch_all_data(ticker, bond_ticker, vix_ticker, days):
    """获取所有所需的市场数据"""
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)

    data = {}
    try:
        # 1. 主要标的 (QQQ/SPY)
        ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if ticker_data.empty:
            st.error(f"无法获取 {ticker} 数据")
            return None
        data['primary'] = ticker_data

        # 2. VIX指数
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date, progress=False)
        data['vix'] = vix_data

        # 3. 垃圾债 ETF (HYG) 和国债 (TLT) 用于计算利差
        hyg_data = yf.download(bond_ticker, start=start_date, end=end_date, progress=False)
        tlt_data = yf.download("TLT", start=start_date, end=end_date, progress=False)  # 20年以上国债ETF作为长期利率代理
        data['hyg'] = hyg_data
        data['tlt'] = tlt_data

        # 4. 用于市场广度的标普500成分股 (用SPY近似代替计算)
        spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)
        data['spy'] = spy_data

        # 5. 用于Put/Call Ratio (由于没有直接免费API，此处使用VIX和SKEW指数估算市场情绪)
        # 注意：实际PCR数据通常需要付费API，此处为模拟逻辑
        data['pcr_estimate'] = (vix_data['Close'] / vix_data['Close'].rolling(20).mean()).to_frame(name='PCR_Estimate')

        return data
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

# 获取数据
with st.spinner('正在从市场获取最新数据...'):
    all_data = fetch_all_data(ticker_symbol, bond_ticker, vix_ticker, lookback_days)

if not all_data:
    st.stop()

primary_df = all_data['primary']
vix_df = all_data['vix']
hyg_df = all_data['hyg']
tlt_df = all_data['tlt']
spy_df = all_data['spy']
pcr_df = all_data['pcr_estimate']

# ----------------------------- 指标计算函数 -----------------------------
def calculate_technical_indicators(df):
    """计算技术指标"""
    df = df.copy()
    close = df['Close']

    # 移动平均线
    df['MA_50'] = talib.SMA(close, timeperiod=50)
    df['MA_100'] = talib.SMA(close, timeperiod=100)
    df['MA_200'] = talib.SMA(close, timeperiod=200)

    # 价格偏离度 (%)
    df['Dev_50'] = (close / df['MA_50'] - 1) * 100
    df['Dev_100'] = (close / df['MA_100'] - 1) * 100
    df['Dev_200'] = (close / df['MA_200'] - 1) * 100

    # 短期动量 (5日收益率)
    df['Momentum_5D'] = close.pct_change(5) * 100

    # ATR (平均真实波幅) 用于衡量波动性
    df['ATR'] = talib.ATR(df['High'], df['Low'], close, timeperiod=14)

    # RSI (相对强弱指数)
    df['RSI'] = talib.RSI(close, timeperiod=14)

    return df

def calculate_market_breadth(df):
    """计算市场广度（简化版：使用上涨/下跌比例）"""
    # 由于实时获取全市场股票数据复杂，此处使用SPY价格与均线关系模拟广度
    df['Breadth_Proxy'] = (df['Close'] > df['Close'].rolling(50).mean()).astype(int) * 100
    return df

def calculate_bond_spread(hyg_df, tlt_df):
    """计算垃圾债利差（简化：使用HYG与TLT的价格比率变化作为代理）"""
    spread_series = (hyg_df['Close'] / tlt_df['Close']).pct_change(5) * 100
    return spread_series.rename('Junk_Spread_Change')

def calculate_yield_curve():
    """收益率曲线（10Y-2Y）代理指标"""
    # 注：实际利差数据需从FRED等API获取，此处使用TLT与IEF的比率变化模拟
    # 这是一个占位逻辑，实际应用中应替换为真实数据
    dates = primary_df.index
    synthetic_yield_spread = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.5 + 0.2  # 模拟波动
    return pd.Series(synthetic_yield_spread, index=dates, name='Yield_Spread_Proxy')

# 执行计算
primary_df = calculate_technical_indicators(primary_df)
primary_df['Breadth'] = calculate_market_breadth(spy_df)['Breadth_Proxy']
primary_df['Junk_Spread'] = calculate_bond_spread(hyg_df, tlt_df)
primary_df['Yield_Curve'] = calculate_yield_curve()

# 确保所有数据长度一致
common_index = primary_df.index
vix_series = vix_df['Close'].reindex(common_index).ffill()
pcr_series = pcr_df['PCR_Estimate'].reindex(common_index).ffill()

# ----------------------------- 综合评分模型 -----------------------------
def calculate_composite_score(row):
    """根据单行数据计算0-100的综合评分"""
    score = 50  # 起始中性分

    # 1. 价格偏离度评分 (权重: 20%)
    dev_avg = (row.get('Dev_50', 0) + row.get('Dev_100', 0) + row.get('Dev_200', 0)) / 3
    if -2 < dev_avg < 5:  # 偏离度适中
        score += 10
    elif dev_avg < -10 or dev_avg > 15:  # 严重偏离
        score -= 15
    else:
        score += (5 - abs(dev_avg)/5)  # 线性调整

    # 2. VIX指数评分 (权重: 20%)
    vix_val = vix_series.get(row.name, 20)
    if vix_val < 16:
        score += 10  # 低波动，市场乐观
    elif vix_val > 30:
        score -= 10  # 高波动，市场恐慌
    else:
        score += (30 - vix_val) / 1.5  # 15-30之间线性评分

    # 3. 市场广度评分 (权重: 15%)
    breadth_val = row.get('Breadth', 50)
    if breadth_val > 70:
        score += 7.5
    elif breadth_val < 30:
        score -= 7.5
    else:
        score += (breadth_val - 50) / 4

    # 4. 垃圾债利差评分 (权重: 15%)
    spread_val = row.get('Junk_Spread', 0)
    if spread_val < -2:  # 利差收窄（风险偏好）
        score += 7.5
    elif spread_val > 5:  # 利差急剧扩大（风险规避）
        score -= 7.5
    else:
        score -= spread_val * 1.5

    # 5. 收益率曲线评分 (权重: 15%)
    yield_val = row.get('Yield_Curve', 0)
    if yield_val > 0.1:
        score += 7.5  # 曲线陡峭，经济预期乐观
    elif yield_val < -0.2:
        score -= 7.5  # 曲线倒挂，衰退预警
    else:
        score += yield_val * 30

    # 6. Put/Call Ratio 评分 (权重: 10%)
    pcr_val = pcr_series.get(row.name, 1.0)
    if pcr_val < 0.8:  # 极端看涨，可能过热
        score -= 5
    elif pcr_val > 1.2:  # 极端看跌，可能超卖
        score += 5
    else:
        score += (1.0 - pcr_val) * 10

    # 7. 短期动量评分 (权重: 5%)
    mom_val = row.get('Momentum_5D', 0)
    if mom_val > 3:
        score += 2.5
    elif mom_val < -3:
        score -= 2.5
    else:
        score += mom_val / 1.5

    # 将分数限制在0-100之间
    return max(0, min(100, score))

# 应用评分模型
primary_df['Composite_Score'] = primary_df.apply(calculate_composite_score, axis=1)

# ----------------------------- 风险识别与噪音过滤 -----------------------------
def identify_risk_signals(row):
    """识别特别风险信号"""
    risks = []
    vix_val = vix_series.get(row.name, 20)
    pcr_val = pcr_series.get(row.name, 1.0)

    # 规则1: 极端波动预警
    if vix_val > 35:
        risks.append(("🔄 极端波动", f"VIX指数高达 {vix_val:.1f}，市场处于极端恐慌状态。", "high"))
    elif vix_val > 25:
        risks.append(("⚠️ 波动升高", f"VIX指数为 {vix_val:.1f}，市场波动性显著增加。", "medium"))

    # 规则2: 广度衰竭 (价格新高但广度未确认)
    if row.name == primary_df.index[-1]:  # 只检查最新数据点
        recent_high = primary_df['Close'].tail(20).max()
        recent_breadth_high = primary_df['Breadth'].tail(20).max()
        current_price = row['Close']
        current_breadth = row['Breadth']
        if current_price >= recent_high * 0.99 and current_breadth < recent_breadth_high * 0.95:
            risks.append(("📉 广度衰竭", "价格接近高点但市场广度未能确认，上涨动力可能不足。", "medium"))

    # 规则3: 均线死亡交叉预警
    if 'MA_50' in row and 'MA_200' in row:
        if row['MA_50'] < row['MA_200'] and primary_df.index.get_loc(row.name) > 0:
            prev_idx = primary_df.index[primary_df.index.get_loc(row.name) - 1]
            if primary_df.loc[prev_idx, 'MA_50'] >= primary_df.loc[prev_idx, 'MA_200']:
                risks.append(("💀 死亡交叉", "50日均线下穿200日均线，长期趋势可能转弱。", "high"))

    # 规则4: PCR极端值
    if pcr_val > 1.3:
        risks.append(("📊 极端看跌情绪", f"Put/Call Ratio估计值高达 {pcr_val:.2f}，市场情绪极度悲观。", "medium"))
    elif pcr_val < 0.6:
        risks.append(("📈 极端看涨情绪", f"Put/Call Ratio估计值低至 {pcr_val:.2f}，市场可能过热。", "medium"))

    # 规则5: 价格严重偏离均线
    dev_200 = row.get('Dev_200', 0)
    if dev_200 > 15:
        risks.append(("🚀 严重超买", f"价格偏离200日均线达 {dev_200:.1f}%，回调风险增加。", "medium"))
    elif dev_200 < -15:
        risks.append(("🏃 严重超卖", f"价格偏离200日均线达 {dev_200:.1f}%，可能出现技术性反弹。", "low"))

    return risks

# ----------------------------- 仪表板布局 -----------------------------
# 顶部关键指标卡片
col1, col2, col3, col4 = st.columns(4)
latest = primary_df.iloc[-1]
latest_score = latest['Composite_Score']

with col1:
    st.metric("综合评分", f"{latest_score:.1f}/100")
with col2:
    score_change = latest_score - primary_df.iloc[-2]['Composite_Score']
    st.metric("评分变化", f"{score_change:+.1f}")
with col3:
    st.metric(f"{ticker_symbol} 价格", f"${latest['Close']:.2f}")
with col4:
    vix_val = vix_series.iloc[-1]
    st.metric("VIX指数", f"{vix_val:.2f}")

# 综合评分趋势图
st.subheader("综合评分趋势")
fig_score = go.Figure()
fig_score.add_trace(go.Scatter(x=primary_df.index, y=primary_df['Composite_Score'],
                               mode='lines', name='综合评分', line=dict(color='royalblue', width=3)))
fig_score.add_hrect(y0=70, y1=100, fillcolor="lightgreen", opacity=0.2, layer="below", annotation_text="乐观区域")
fig_score.add_hrect(y0=30, y1=70, fillcolor="lightyellow", opacity=0.2, layer="below", annotation_text="中性区域")
fig_score.add_hrect(y0=0, y1=30, fillcolor="lightcoral", opacity=0.2, layer="below", annotation_text="谨慎区域")
fig_score.update_layout(height=400, yaxis_range=[0, 100], hovermode='x unified')
st.plotly_chart(fig_score, use_container_width=True)

# 分项指标图表
st.subheader("分项指标监控")
fig_indicators = make_subplots(
    rows=3, cols=2,
    subplot_titles=('价格与移动平均线', 'VIX恐慌指数', '价格偏离度 (%)', '市场广度', '垃圾债利差变化', '收益率曲线代理'),
    vertical_spacing=0.12
)

# 图1: 价格与MA
fig_indicators.add_trace(
    go.Scatter(x=primary_df.index, y=primary_df['Close'], name='价格', line=dict(color='black')),
    row=1, col=1
)
for ma, color in [('MA_50', 'blue'), ('MA_100', 'orange'), ('MA_200', 'red')]:
    if ma in primary_df.columns:
        fig_indicators.add_trace(
            go.Scatter(x=primary_df.index, y=primary_df[ma], name=ma, line=dict(color=color, dash='dash')),
            row=1, col=1
        )

# 图2: VIX
fig_indicators.add_trace(
    go.Scatter(x=vix_series.index, y=vix_series, name='VIX', line=dict(color='purple')),
    row=1, col=2
)
fig_indicators.add_hline(y=20, line=dict(color='gray', dash='dash'), row=1, col=2)

# 图3: 价格偏离度
for dev, color in [('Dev_50', 'lightblue'), ('Dev_100', 'lightgreen'), ('Dev_200', 'lightsalmon')]:
    if dev in primary_df.columns:
        fig_indicators.add_trace(
            go.Scatter(x=primary_df.index, y=primary_df[dev], name=dev, line=dict(color=color)),
            row=2, col=1
        )
fig_indicators.add_hline(y=0, line=dict(color='gray', dash='dash'), row=2, col=1)

# 图4: 市场广度
if 'Breadth' in primary_df.columns:
    fig_indicators.add_trace(
        go.Scatter(x=primary_df.index, y=primary_df['Breadth'], name='广度', line=dict(color='green')),
        row=2, col=2
    )
    fig_indicators.add_hline(y=50, line=dict(color='gray', dash='dash'), row=2, col=2)

# 图5: 垃圾债利差变化
if 'Junk_Spread' in primary_df.columns:
    fig_indicators.add_trace(
        go.Bar(x=primary_df.index, y=primary_df['Junk_Spread'], name='垃圾债利差', marker_color='coral'),
        row=3, col=1
    )
    fig_indicators.add_hline(y=0, line=dict(color='gray', dash='dash'), row=3, col=1)

# 图6: 收益率曲线代理
if 'Yield_Curve' in primary_df.columns:
    fig_indicators.add_trace(
        go.Scatter(x=primary_df.index, y=primary_df['Yield_Curve'], name='收益率曲线', line=dict(color='brown')),
        row=3, col=2
    )
    fig_indicators.add_hline(y=0, line=dict(color='gray', dash='dash'), row=3, col=2)

fig_indicators.update_layout(height=900, showlegend=True, hovermode='x unified')
st.plotly_chart(fig_indicators, use_container_width=True)

# ----------------------------- 风险提示面板 -----------------------------
st.subheader("🔔 风险与信号提示")
latest_risks = identify_risk_signals(primary_df.iloc[-1])

if latest_risks:
    for risk_title, risk_desc, risk_level in latest_risks:
        if risk_level == "high":
            st.error(f"**{risk_title}**: {risk_desc}")
        elif risk_level == "medium":
            st.warning(f"**{risk_title}**: {risk_desc}")
        else:
            st.info(f"**{risk_title}**: {risk_desc}")
else:
    st.success("当前未检测到高风险信号。市场状况处于正常波动范围内。")

# 最新数据表格
with st.expander("查看最近5个交易日的数据快照"):
    display_cols = ['Close', 'Composite_Score', 'Dev_200', 'Momentum_5D', 'Breadth']
    available_cols = [col for col in display_cols if col in primary_df.columns]
    st.dataframe(primary_df[available_cols].tail().round(2))

# ----------------------------- 页脚 -----------------------------
st.markdown("---")
st.caption("""
**数据说明与免责声明**:
- 数据来源: Yahoo Finance，可能存在15-20分钟延迟。
- 垃圾债利差使用HYG/TLT比率变化代理；收益率曲线为模拟数据，真实应用中需接入FRED API。
- Put/Call Ratio为基于VIX的估算值。
- **本仪表板仅为技术分析工具，不构成任何投资建议。市场有风险，决策需谨慎。**
""")