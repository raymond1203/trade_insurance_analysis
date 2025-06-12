# prediction_utils.py
# 통합 예측 모델, 피처 엔지니어링, 시나리오 예측 함수 모음

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def create_lag_features(export_data):
    """
    수출 데이터를 기반으로 시차 피처 생성
    
    export_data: 수출입 데이터
    return: 시차 기반 피처
    """
    try:
        # 국가별로 시차 피처 생성
        lag_features = []
        
        for country in export_data['국가'].unique():
            country_data = export_data[export_data['국가'] == country].copy()
            country_data = country_data.sort_values('년월')
            
            # 기본 피처
            country_features = {
                '국가': country,
                '총수출액': country_data['수출액'].sum(),
                '평균수출액': country_data['수출액'].mean(),
                '수출변동성': country_data['수출액'].std(),
                '최대수출액': country_data['수출액'].max(),
                '최소수출액': country_data['수출액'].min()
            }
            
            # 시차 피처 (6, 12, 18개월)
            for lag in [6, 12, 18]:
                if len(country_data) > lag:
                    lagged_export = country_data['수출액'].shift(lag)
                    country_features[f'수출액_lag_{lag}'] = lagged_export.mean()
                    country_features[f'수출증가율_lag_{lag}'] = ((country_data['수출액'] - lagged_export) / lagged_export * 100).mean()
            
            # 계절성 피처 (월별 평균)
            country_data['월'] = pd.to_datetime(country_data['년월']).dt.month
            monthly_avg = country_data.groupby('월')['수출액'].mean()
            country_features['계절성_변동계수'] = monthly_avg.std() / monthly_avg.mean()
            
            # 트렌드 피처 (연도별 증가율)
            yearly_data = country_data.groupby('연도')['수출액'].sum()
            if len(yearly_data) > 1:
                trend = np.polyfit(range(len(yearly_data)), yearly_data.values, 1)[0]
                country_features['연평균_증가추세'] = trend
            else:
                country_features['연평균_증가추세'] = 0
            
            lag_features.append(country_features)
        
        return pd.DataFrame(lag_features)
    
    except Exception as e:
        print(f"시차 피처 생성 오류: {e}")
        return pd.DataFrame()

def create_ai_features(risk_index_data):
    """
    AI 위험지수 데이터를 기반으로 피처 생성
    
    risk_index_data: AI 위험지수 데이터
    return: AI 기반 피처
    """
    try:
        # 국가별 위험지수 집계
        country_risk = risk_index_data.groupby('국가명').agg({
            '위험지수': ['mean', 'max', 'min', 'std', 'count']
        }).round(4)
        
        # 컬럼명 정리
        country_risk.columns = ['평균위험지수', '최대위험지수', '최소위험지수', '위험지수변동성', '업종수']
        country_risk = country_risk.reset_index()
        country_risk.rename(columns={'국가명': '국가'}, inplace=True)
        
        # 위험등급 분류
        country_risk['위험등급'] = pd.cut(country_risk['평균위험지수'], 
                                       bins=[0, 2, 3.5, 5], 
                                       labels=[1, 2, 3])  # 1=저위험, 2=중위험, 3=고위험
        
        # 업종별 위험 분산도
        country_risk['위험분산도'] = country_risk['위험지수변동성'] / country_risk['평균위험지수']
        
        return country_risk
    
    except Exception as e:
        print(f"AI 피처 생성 오류: {e}")
        return pd.DataFrame()

def create_interaction_features(lag_features, ai_features):
    """
    시차 피처와 AI 피처 간의 상호작용 피처 생성
    
    return: 상호작용 피처
    """
    try:
        # 데이터 병합
        merged_data = pd.merge(lag_features, ai_features, on='국가', how='inner')
        
        # 상호작용 피처 생성
        merged_data['수출액_위험지수_비율'] = merged_data['총수출액'] / (merged_data['평균위험지수'] + 1)
        merged_data['변동성_위험지수_곱'] = merged_data['수출변동성'] * merged_data['평균위험지수']
        merged_data['트렌드_위험등급_곱'] = merged_data['연평균_증가추세'] * merged_data['위험등급'].astype(float)
        
        # 정규화 피처
        merged_data['수출액_정규화'] = (merged_data['총수출액'] - merged_data['총수출액'].min()) / \
                                   (merged_data['총수출액'].max() - merged_data['총수출액'].min())
        merged_data['위험지수_정규화'] = (merged_data['평균위험지수'] - merged_data['평균위험지수'].min()) / \
                                     (merged_data['평균위험지수'].max() - merged_data['평균위험지수'].min())
        
        return merged_data
    
    except Exception as e:
        print(f"상호작용 피처 생성 오류: {e}")
        return pd.DataFrame()

def build_ensemble_model(X, y):
    """
    앙상블 모델 구축
    
    X: 입력 피처
    y: 타겟
    return: 앙상블 모델 객체
    """
    try:
        # 데이터 전처리
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        y_clean = pd.Series(y).fillna(0)
        
        if len(X_clean) == 0 or len(y_clean) == 0:
            return None, {}
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42
        )
        
        # 개별 모델들
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        
        # 앙상블 모델
        ensemble_model = VotingRegressor([
            ('rf', rf_model),
            ('xgb', xgb_model), 
            ('lgb', lgb_model)
        ])
        
        # 모델 훈련
        ensemble_model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = ensemble_model.predict(X_test)
        
        performance = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return ensemble_model, performance
    
    except Exception as e:
        print(f"앙상블 모델 구축 오류: {e}")
        return None, {}

def predict_scenarios(model, base_data, scenarios):
    """
    시나리오별 예측
    
    model: 학습된 모델
    base_data: 기준 데이터
    scenarios: 시나리오 dict
    return: 시나리오별 예측 결과
    """
    try:
        if model is None:
            return {}
        
        scenario_results = {}
        
        # 기준 데이터 준비
        base_features = base_data.select_dtypes(include=[np.number]).fillna(0)
        
        for scenario_name, scenario_desc in scenarios.items():
            scenario_data = base_features.copy()
            
            # 시나리오별 데이터 조정
            if scenario_name == 'optimistic':
                # 낙관적: 수출 10% 증가, 위험지수 10% 감소
                export_cols = [col for col in scenario_data.columns if '수출' in col]
                risk_cols = [col for col in scenario_data.columns if '위험' in col]
                
                for col in export_cols:
                    scenario_data[col] = scenario_data[col] * 1.1
                for col in risk_cols:
                    scenario_data[col] = scenario_data[col] * 0.9
                    
            elif scenario_name == 'pessimistic':
                # 비관적: 수출 20% 감소, 위험지수 20% 증가
                export_cols = [col for col in scenario_data.columns if '수출' in col]
                risk_cols = [col for col in scenario_data.columns if '위험' in col]
                
                for col in export_cols:
                    scenario_data[col] = scenario_data[col] * 0.8
                for col in risk_cols:
                    scenario_data[col] = scenario_data[col] * 1.2
                    
            # 예측 수행
            predictions = model.predict(scenario_data)
            
            scenario_results[scenario_name] = {
                'description': scenario_desc,
                'predictions': predictions.tolist(),
                'mean_prediction': float(np.mean(predictions)),
                'std_prediction': float(np.std(predictions)),
                'max_prediction': float(np.max(predictions)),
                'min_prediction': float(np.min(predictions))
            }
        
        return scenario_results
    
    except Exception as e:
        print(f"시나리오 예측 오류: {e}")
        return {}

def feature_importance_analysis(model, feature_names):
    """
    피처 중요도 분석
    
    model: 훈련된 모델
    feature_names: 피처명 리스트
    return: 피처 중요도 결과
    """
    try:
        # VotingRegressor의 경우 개별 모델들의 중요도 평균
        if hasattr(model, 'estimators_'):
            importances = []
            
            for estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                avg_importance = np.mean(importances, axis=0)
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
                
                return feature_importance
        
        return pd.DataFrame()
    
    except Exception as e:
        print(f"피처 중요도 분석 오류: {e}")
        return pd.DataFrame()

def calculate_prediction_intervals(model, X, confidence=0.95):
    """
    예측 구간 계산
    
    model: 훈련된 모델
    X: 입력 데이터
    confidence: 신뢰도
    return: 예측 구간
    """
    try:
        # 부트스트랩을 통한 예측 구간 계산
        n_bootstrap = 100
        predictions = []
        
        for i in range(n_bootstrap):
            # 부트스트랩 샘플링
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[indices]
            
            # 예측
            pred = model.predict(X_bootstrap)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 신뢰구간 계산
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        mean_prediction = np.mean(predictions, axis=0)
        
        return {
            'mean': mean_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidence
        }
    
    except Exception as e:
        print(f"예측 구간 계산 오류: {e}")
        return {} 