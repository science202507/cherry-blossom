import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- [1] 웹 페이지 설정 및 한글 폰트 ---
st.set_page_config(page_title="벚꽃 개화 예측 AI", layout="wide")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False 

st.title("🌸 벚꽃 개화 예측 AI 대시보드")

# --- [2] 데이터 로드 엔진 ---
@st.cache_data
def load_data():
    try:
        df_b = pd.read_csv('blossoms.csv', encoding='cp949', skiprows=2, header=None)
        df_b.columns = ['지점', '년도', '발아', '발아_평비', '개화', '개화_평비', '만발', '만발_평비']
        df_b['개화일_dt'] = pd.to_datetime(df_b['개화'], errors='coerce')
        df_b = df_b.dropna(subset=['개화일_dt'])
        df_b['개화일_숫자'] = df_b['개화일_dt'].dt.dayofyear
        
        df_w = pd.read_csv('avgtemp_rain_sun.csv', encoding='cp949')
        df_w.columns = df_w.columns.str.strip()
        if '지점' in df_w.columns: df_w = df_w.drop(columns=['지점'])
        df_w = df_w.rename(columns={'지점명': '지점', '평균기온(°C)': 'temp'})
        df_w = df_w[df_w['일시'].str.contains('-03', na=False)].copy()
        df_w['년도'] = df_w['일시'].str[:4].astype(int)
        
        return pd.merge(df_b[['지점', '년도', '개화일_숫자']], 
                        df_w[['지점', '년도', 'temp']], on=['지점', '년도']).dropna()
    except Exception:
        return pd.DataFrame()

# --- [3] 메인 화면 구현 ---
data = load_data()

if not data.empty:
    target_city = st.sidebar.selectbox("📍 지역 선택", sorted(data['지점'].unique()))
    city_df = data[data['지점'] == target_city].sort_values('년도')

    X_ols = sm.add_constant(city_df['temp'])
    model = sm.OLS(city_df['개화일_숫자'], X_ols).fit()

    # 레이아웃 구성
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader(f"📊 {target_city} 통계 요약")
        st.write(f"**회귀 방정식:** $y = {model.params['temp']:.2f}x + {model.params['const']:.2f}$")
        st.write(f"**결정계수 ($R^2$):** {model.rsquared:.3f}")
        
        # [수정 지점] rf 접두사와 이중 중괄호 사용
        st.info(rf"💡 기온 $1^\circ\text{{C}}$ 오를 때마다 약 {abs(model.params['temp']):.1f}일 일찍 핍니다.")

    with col2:
        st.subheader("📈 기온-개화일 상관관계")
        fig, ax = plt.subplots()
        sns.regplot(data=city_df, x='temp', y='개화일_숫자', ax=ax, color='pink', line_kws={"color": "red"})
        ax.set_xlabel("3월 평균 기온 (°C)")
        ax.set_ylabel("개화일 (1월 1일 기준 날짜)")
        st.pyplot(fig)

    # 하단 예측 섹션
    st.divider()
    st.subheader("🔮 2026년 예측 결과")
    temp_trend = LinearRegression().fit(city_df[['년도']], city_df['temp'])
    pred_temp = temp_trend.predict(pd.DataFrame({'년도': [2026]}))[0]
    pred_day = model.predict([1, pred_temp])[0]
    res_date = datetime(2026, 1, 1) + timedelta(days=int(pred_day)-1)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("2026 예상 기온", f"{pred_temp:.2f} °C")
    c2.metric("2026 예상 개화일", res_date.strftime('%m월 %d일'))
    c3.metric("모델 설명력", f"{model.rsquared*100:.1f}%")
