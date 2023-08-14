import pandas as pd
import streamlit as st
import pygwalker as pyg

st.set_page_config(
    page_icon=":goose:",
    page_title="부지런한 거위들",
    layout="wide",
)

st.header("""
데이터셋 개요
""")

tab1, tab2, tab3 = st.tabs(["전복나이예측", "펄서여부예측", "스테인레스결함예측"])

with tab1: 
    st.subheader('전복나이예측 (Regression)')
    st.image('./img/abalone.png', width=400)
    
    st.write("""
    #### 1. 개요
    """)
    
    st.write("""
    이 데이터 세트에는 4177개의 행과 9개의 열이 있으며 
    전복의 물리적 측정값과 나이테 수(연령을 나타냄)가 있습니다.     
    전복의 나이를 확인하려면 전복의 뿔을 가로질러 껍질을 자르고 염색한 다음 
    현미경으로 고리의 수를 세야 합니다.     
    이는 지루하고 시간이 많이 걸리는 작업입니다.     
    이 데이터셋은 보다 쉽게 전복의 나이를 예측하는데 사용하기 위함입니다.     
    """)

     
    st.write("""
    #### 2. 데이터셋 설명
    """)

    st.write("""
    이 데이터셋은 조개의 여러 특성에 관한 정보를 제공합니다. 각 열의 의미는 다음과 같습니다:  

    1. **Sex**: 조개의 성별을 나타냅니다. 'M'은 수컷, 'F'는 암컷, 'I'는 유아기를 나타냅니다.  
    2. **Length**: 조개의 가장 긴 부분의 길이를 나타냅니다.  
    3. **Diameter**: 수직으로 측정한 조개의 직경을 나타냅니다.  
    4. **Height**: 조개의 높이를 나타냅니다(조개껍질 포함).  
    5. **Whole weight**: 조개 전체의 무게를 나타냅니다.
    6. **Shucked weight**: 조개의 살만을 제거한 무게를 나타냅니다.  
    7. **Viscera weight**: 조개의 내장 무게를 나타냅니다.  
    8. **Shell weight**: 조개껍질의 무게를 나타냅니다.  
    9. **Rings**: 조개의 나이를 나타냅니다. 일반적으로, 링의 개수에 1.5를 더하면 조개의 나이(년)가 됩니다.  

    각 행은 특정 조개의 위 특성을 나타냅니다.  
    이 데이터를 사용하여 조개의 특성과 나이 사이의 관계를 분석할 수 있습니다.  
    예를 들어, 회귀 분석을 사용하여 특정 특성(예: 무게, 길이 등)이 조개의 나이에 어떤 영향을 미치는지 예측할 수 있습니다.  
    """)
    
    st.write("""
    #### 3. 데이터셋 확인
    """)
    data = pd.read_csv('./csv/Regression_data.csv')
    walker = pyg.walk(data, env='Streamlit')

with tab2: 
    st.subheader('펄서여부예측 (binary_classification)')
    st.image('./img/pulsar.jpg', width=400)

    st.write("""
    #### 1. 개요
    """)
    
    st.write("""
    이 데이터셋의 목적은 천체학 데이터를 기반으로 펄서를 식별하는 머신러닝 모델을 만드는 것입니다.  
    HTRU2는 고시간 해상도 우주 조사에서 수집된 펄서 후보의 샘플을 설명하는 데이터 세트입니다.  
    펄서는 지구에서 감지할 수 있는 전파를 방출하는 희귀한 유형의 중성자 별입니다.   
    펄서는 시공간, 항성 간 매체 및 물질 상태에 대한 탐사선으로서 과학적으로 상당한 관심을 받고 있습니다.  
    펄서는 자전하면서 방출 빔이 하늘을 가로지르고, 이것이 우리의 가시선을 넘어가면 감지 가능한 광대역 전파 방출 패턴을 생성합니다.   
    펄서는 빠르게 회전하기 때문에 이 패턴이 주기적으로 반복됩니다.   
    따라서 펄서 탐색에는 대형 전파 망원경으로 주기적인 전파 신호를 찾는 것이 포함됩니다.  
    각 펄서는 약간씩 다른 방출 패턴을 생성하며, 이는 회전할 때마다 조금씩 달라집니다.   
    따라서 '후보'로 알려진 잠재적 신호 검출은 관측 길이에 따라 결정되는 펄서의 여러 회전에 걸쳐 평균화됩니다.   
    그러나 거의 모든 탐지가 무선 주파수 간섭(RFI)과 잡음으로 인해 발생하기 때문에 정상적인 신호를 찾기가 어렵습니다.  
    이에 머신 러닝 도구를 사용하여 펄서 후보에 대한 신속한 분석을 하고 있습니다.   
    특히 후보 데이터 세트를 이진 분류 문제로 처리하는 분류 시스템이 널리 채택되고 있습니다.   
    """)

     
    st.write("""
    #### 2. 데이터셋 설명
    """)

    st.write("""
    이 데이터셋은 특정 천체 유형인 펄서에 대한 관측 데이터를 제공합니다.   
    펄서는 매우 빠르게 회전하는 중성자 별로 고유한 방식으로 신호를 방출합니다.   
    이 데이터셋의 각 행은 하나의 펄서 관측을 나타내며, 각 열은 다음과 같은 특성을 나타냅니다:  

    1. **Mean of the integrated profile**: 펄서 신호의 단일 펄스 프로파일의 평균을 측정합니다.   
       펄서는 고유한 펄스 프로파일을 가지며, 이는 펄서의 자체적인 "지문"과 같습니다.   
       이 값이 높으면 펄서 신호가 더 강력하다는 것을 의미할 수 있습니다.  

    2. **Standard deviation of the integrated profile**: 통합 프로파일의 변동성을 나타냅니다.   
        펄서 신호의 강도가 일정하게 유지되는 경우 표준 편차는 작을 것입니다.   
        더 큰 표준 편차는 신호가 더 변동성이 있음을 나타냅니다.  
  
    3. **Excess kurtosis of the integrated profile**: 첨도는 분포의 뾰족함을 측정합니다.   
       펄서의 펄스 분포가 정규 분포보다 뾰족한 경우(즉, 신호값이 평균 근처에 더 집중되어 있는 경우), 이 값은 높아질 것입니다.  
    4. **Skewness of the integrated profile**: 이 값은 신호의 분포가 얼마나 비대칭적인지를 측정합니다.   
       왜도가 0이 아닌 값은 분포가 왼쪽이나 오른쪽으로 치우쳐 있다는 것을 의미합니다.   
    펄서의 신호가 일반적으로 어떤 방향으로 치우쳐 있을 경우, 이는 펄서의 회전이나 에너지 방출 패턴에 대한 중요한 정보를 제공할 수 있습니다.  

    5. **Mean of the DM-SNR curve**: 이 값은 Dispersion Measure (DM)와 Signal-to-Noise Ratio (SNR) 곡선의 평균 값을 나타냅니다.   
       SNR이 높으면 신호가 잡음에 비해 더 크다는 것을 의미하며, 이는 신호를 더 쉽게 감지하고 이해할 수 있음을 나타냅니다.   
       따라서, DM-SNR 곡선의 평균이 높을수록 펄서 신호를 더 잘 감지할 수 있을 것입니다.  
    6. **Standard deviation of the DM-SNR curve**: 이 값은 DM-SNR 곡선의 변동성을 나타냅니다.   
       변동성이 높을수록 신호는 더 다양한 값을 가질 수 있습니다.  
       이는 DM-SNR 곡선이 많은 정보를 포함하고 있음을 나타낼 수 있지만, 동시에 신호가 더 복잡하고 해석하기 어려울 수 있다는 것을 의미합니다.  
    7. **Excess kurtosis of the DM-SNR curve**: 이 값은 DM-SNR 곡선의 첨도를 나타냅니다.  
        첨도가 높을 경우, DM-SNR 값들이 특정 값들 주변에 집중되어 있다는 것을 의미합니다.   
        이는 펄서 신호가 일정한 패턴을 가질 가능성이 높음을 나타내며, 이 정보는 펄서와 비펄서를 구별하는 데 도움이 될 수 있습니다.  
     8. **Skewness of the DM-SNR curve**: 이 값은 DM-SNR 곡선의 왜도를 나타냅니다.   
        왜도가 0에서 벗어난다면, 이는 DM-SNR 값들이 특정 방향으로 치우쳐 있다는 것을 의미합니다.   
        이 정보는 펄서 신호의 발생 패턴이나 방출 방향 등에 대한 중요한 힌트를 제공할 수 있습니다.  
    9. **target_class**: 이것은 우리가 예측하려는 목표 변수입니다. 1은 펄서를 나타내고 0은 펄서가 아닌 것을 나타냅니다.  

    이 모든 특성들은 신호의 다양한 측면을 나타내며, 이들을 함께 사용하여 우리는 펄서인지 아닌지를 분류하거나 예측할 수 있습니다.  
    각 특성은 독립적으로 펄서를 판별하지 않지만, 함께 사용하면 머신러닝 알고리즘이 펄서를 식별하는 데 사용할 수 있는 유용한 패턴을 찾을 수 있습니다.  
    """)
    
    st.write("""
    #### 3. 데이터셋 확인
    """)
    data = pd.read_csv('./csv/binary_classification_data.csv')
    walker = pyg.walk(data, env='Streamlit')

with tab3:
    st.subheader('스테인레스결함데이터 (multi_classification)')
    st.image('./img/stainless.jpg', width=400)
    
    st.write("""
    #### 1. 개요
    """)
    
    st.write("""
    이 데이터셋은 Semeion, 통신과학 연구센터에서 연구한 결과입니다.   
    원래의 연구 목표는 스테인레스 스틸 판의 표면 결함 유형을 올바르게 분류하는 것이었으며,   
    가능한 결함 유형은 6가지(그 외 "기타" 포함)였습니다.   
    입력 벡터는 결함의 기하학적 형태와 윤곽선을 대략적으로 설명하는 27개의 지표로 구성되었습니다.   
    연구 논문에 따르면, 이 작업은 Centro Sviluppo Materiali (이탈리아)로부터 의뢰받은 것이므로,   
    입력 벡터로 사용된 27개의 지표의 성격이나 6개의 결함 클래스의 유형에 대한 자세한 내용을 알 수 없습니다.  
    """)

     
    st.write("""
    #### 2. 데이터셋 설명
    """)

    st.write("""
    필드는 총 34개입니다.     
    처음 27개의 필드는 이미지에서 볼 수 있는 어떤 종류의 스틸 판 결함을 설명합니다.    
    불행히도, 이 열들을 설명할 수 있는 다른 정보는 없습니다.    

    - 독립변수 : 1열 ~ 27열  
    - 종속변수 : 28열 ~ 34열 (Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults)  

이 데이터셋의 각 열(column)은 아래와 같은 정보를 포함하고 있습니다:

    1. **X_Minimum, X_Maximum, Y_Minimum, Y_Maximum**: 이들은 각 결함의 위치를 나타내는 픽셀 좌표입니다.   
    2. **Pixels_Areas**: 결함 영역의 픽셀 수입니다.  
    3. **X_Perimeter, Y_Perimeter**: 결함 영역의 X축과 Y축을 따라 픽셀의 둘레입니다.  
    4. **Sum_of_Luminosity**: 결함 영역 내의 모든 픽셀의 밝기 합입니다.  
    5. **Minimum_of_Luminosity, Maximum_of_Luminosity**: 결함 영역 내의 최소 및 최대 픽셀 밝기입니다.  
    6. **Length_of_Conveyer**: 컨베이어 벨트의 길이입니다.  
    7. **TypeOfSteel_A300, TypeOfSteel_A400**: 강철의 유형을 나타내는 이진 변수입니다.  
    8. **Steel_Plate_Thickness**: 강판의 두께입니다.  
    9. **Edges_Index, Empty_Index, Square_Index, Outside_X_Index, Edges_X_Index, Edges_Y_Index, Outside_Global_Index, LogOfAreas, Log_X_Index, Log_Y_Index, Orientation_Index, Luminosity_Index, SigmoidOfAreas**: 이들은 이미지 분석을 통해 계산된 다양한 지표들입니다. 이들은 각 결함의 모양, 크기, 밝기 등에 관한 정보를 제공합니다.  
    10. **Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults**: 이들은 결함의 유형을 나타내는 이진 변수입니다. 각 결함 유형에 해당하면 1, 그렇지 않으면 0입니다.  
    """)
    
    st.write("""
    #### 3. 데이터셋 확인
    """)
    data = pd.read_csv('./csv/multi_classification_data.csv')
    walker = pyg.walk(data, env='Streamlit')
