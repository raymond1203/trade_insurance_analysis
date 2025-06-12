# model_validation.py
# AI 위험지수 예측력 검증, 오류 분석, 개선점 도출 함수 모음

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def calculate_prediction_accuracy(predicted, actual):
    """
    AI 위험지수와 실제 보상 데이터 간의 예측 정확도 계산
    
    predicted: AI 위험지수 예측값 (DataFrame)
    actual: 실제 사고/보상 데이터 (DataFrame)
    return: 정확도 등 평가 지표
    """
    results = {}
    
    try:
        # 국가별 위험지수와 실제 보상 비교
        # 위험지수 평균 계산 (국가별)
        predicted_agg = predicted.groupby('국가명')['위험지수'].agg(['mean', 'max', 'std']).reset_index()
        predicted_agg.columns = ['국가명', '평균위험지수', '최대위험지수', '위험지수표준편차']
        
        # 실제 보상 집계 (국가별)
        actual_agg = actual.groupby('국가명').agg({
            '보상금': ['sum', 'mean'],
            '회수금': ['sum', 'mean'],
            '보상률': 'mean'
        }).reset_index()
        actual_agg.columns = ['국가명', '총보상금', '평균보상금', '총회수금', '평균회수금', '평균보상률']
        
        # 데이터 병합
        merged_data = pd.merge(predicted_agg, actual_agg, on='국가명', how='inner')
        
        if len(merged_data) == 0:
            return {'error': '매칭되는 국가가 없습니다'}
        
        # 위험지수와 실제 손실 간 상관관계
        risk_loss_corr, risk_loss_p = pearsonr(merged_data['평균위험지수'], 
                                               merged_data['총보상금'])
        
        # 위험지수와 보상률 간 상관관계  
        risk_rate_corr, risk_rate_p = pearsonr(merged_data['평균위험지수'], 
                                               merged_data['평균보상률'])
        
        # 위험 등급화 (1-5 → 고/중/저 위험)
        merged_data['위험등급'] = pd.cut(merged_data['평균위험지수'], 
                                       bins=[0, 2, 3.5, 5], 
                                       labels=['저위험', '중위험', '고위험'])
        
        # 실제 손실 등급화 (상위 33% = 고손실, 하위 33% = 저손실)
        loss_33 = merged_data['총보상금'].quantile(0.33)
        loss_67 = merged_data['총보상금'].quantile(0.67)
        merged_data['손실등급'] = pd.cut(merged_data['총보상금'], 
                                       bins=[-np.inf, loss_33, loss_67, np.inf], 
                                       labels=['저손실', '중손실', '고손실'])
        
        # 분류 정확도 계산 (Categorical 비교 문제 해결)
        valid_data = merged_data.dropna(subset=['위험등급', '손실등급'])
        if len(valid_data) > 0:
            # Categorical을 문자열로 변환하여 비교
            risk_labels = valid_data['위험등급'].astype(str)
            loss_labels = valid_data['손실등급'].astype(str)
            classification_accuracy = (risk_labels == loss_labels).mean()
        else:
            classification_accuracy = 0
        
        # 결과 정리
        results = {
            '분석대상국가수': len(merged_data),
            '위험지수_손실금액_상관계수': round(risk_loss_corr, 4),
            '위험지수_손실금액_p값': round(risk_loss_p, 4),
            '위험지수_보상률_상관계수': round(risk_rate_corr, 4),
            '위험지수_보상률_p값': round(risk_rate_p, 4),
            '분류정확도': round(classification_accuracy, 4),
            '등급별_분포': merged_data['위험등급'].astype(str).value_counts().to_dict(),
            '손실등급별_분포': merged_data['손실등급'].astype(str).value_counts().to_dict()
        }
        
        # 상세 데이터도 함께 반환
        results['상세데이터'] = merged_data
        
    except Exception as e:
        results = {'error': str(e)}
    
    return results

def analyze_prediction_errors(predicted, actual, group_by):
    """
    예측 오류 패턴 분석
    
    predicted: 예측값 (DataFrame)
    actual: 실제값 (DataFrame) 
    group_by: ['country', 'sector'] 등
    return: 오류 패턴 분석 결과
    """
    error_analysis = {}
    
    try:
        # 국가별 위험지수와 실제 결과 병합
        predicted_agg = predicted.groupby('국가명')['위험지수'].mean().reset_index()
        actual_agg = actual.groupby('국가명')['보상금'].sum().reset_index()
        
        merged = pd.merge(predicted_agg, actual_agg, on='국가명', how='inner')
        
        if len(merged) == 0:
            return {'error': '분석할 데이터가 없습니다'}
        
        # 정규화 (0-1 스케일)
        merged['위험지수_정규화'] = (merged['위험지수'] - merged['위험지수'].min()) / \
                                  (merged['위험지수'].max() - merged['위험지수'].min())
        merged['보상금_정규화'] = (merged['보상금'] - merged['보상금'].min()) / \
                                (merged['보상금'].max() - merged['보상금'].min())
        
        # 예측 오차 계산
        merged['절대오차'] = abs(merged['위험지수_정규화'] - merged['보상금_정규화'])
        merged['제곱오차'] = (merged['위험지수_정규화'] - merged['보상금_정규화']) ** 2
        
        # 오버/언더 예측 분류
        merged['예측편향'] = merged['위험지수_정규화'] - merged['보상금_정규화']
        merged['예측유형'] = merged['예측편향'].apply(
            lambda x: '과대예측' if x > 0.1 else ('과소예측' if x < -0.1 else '적정예측')
        )
        
        # 그룹별 오류 분석
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # 국가별 오류 패턴
        country_errors = merged.groupby('국가명').agg({
            '절대오차': ['mean', 'std'],
            '제곱오차': 'mean',
            '예측편향': 'mean'
        }).round(4)
        
        # 예측 유형별 분포
        prediction_type_dist = merged['예측유형'].value_counts()
        
        # 최대 오차 국가들
        top_error_countries = merged.nlargest(5, '절대오차')[['국가명', '절대오차', '예측유형']]
        
        # 전체 오류 통계
        overall_stats = {
            '평균절대오차': round(merged['절대오차'].mean(), 4),
            '평균제곱오차': round(merged['제곱오차'].mean(), 4),
            '평균편향': round(merged['예측편향'].mean(), 4),
            '예측정확도': round((merged['예측유형'] == '적정예측').mean(), 4)
        }
        
        error_analysis = {
            '전체통계': overall_stats,
            '국가별오류': country_errors.to_dict(),
            '예측유형분포': prediction_type_dist.to_dict(),
            '최대오차국가': top_error_countries.to_dict('records'),
            '상세데이터': merged
        }
        
    except Exception as e:
        error_analysis = {'error': str(e)}
    
    return error_analysis

def identify_blind_spots(error_analysis):
    """
    AI 모델의 취약점(Blind Spots) 식별
    
    error_analysis: 오류 분석 결과
    return: 개선이 필요한 영역
    """
    blind_spots = {}
    
    try:
        if 'error' in error_analysis:
            return {'error': error_analysis['error']}
        
        merged_data = error_analysis['상세데이터']
        
        # 1. 고위험 국가 중 과소예측 사례
        high_loss_countries = merged_data[merged_data['보상금_정규화'] > 0.7]
        under_predicted_high_risk = high_loss_countries[high_loss_countries['예측유형'] == '과소예측']
        
        # 2. 저위험 국가 중 과대예측 사례  
        low_loss_countries = merged_data[merged_data['보상금_정규화'] < 0.3]
        over_predicted_low_risk = low_loss_countries[low_loss_countries['예측유형'] == '과대예측']
        
        # 3. 예측 변동성이 큰 국가들
        if '국가별오류' in error_analysis:
            high_variance_countries = []
            for country, stats in error_analysis['국가별오류'].items():
                if ('절대오차', 'std') in stats and stats[('절대오차', 'std')] > 0.2:
                    high_variance_countries.append(country)
        
        # 4. 개선 권고사항
        recommendations = []
        
        if len(under_predicted_high_risk) > 0:
            recommendations.append({
                '문제영역': '고손실 국가 과소예측',
                '해당국가': under_predicted_high_risk['국가명'].tolist(),
                '권고사항': '고위험 신호 감지 알고리즘 보강 필요'
            })
        
        if len(over_predicted_low_risk) > 0:
            recommendations.append({
                '문제영역': '저손실 국가 과대예측',
                '해당국가': over_predicted_low_risk['국가명'].tolist(),
                '권고사항': '안전 국가 판별 기준 정교화 필요'
            })
        
        if len(high_variance_countries) > 0:
            recommendations.append({
                '문제영역': '예측 변동성 과다',
                '해당국가': high_variance_countries,
                '권고사항': '예측 안정성 향상을 위한 모델 튜닝 필요'
            })
        
        # 전체 모델 성능 평가
        accuracy = error_analysis['전체통계']['예측정확도']
        if accuracy < 0.6:
            model_grade = '개선필요'
        elif accuracy < 0.8:
            model_grade = '보통'
        else:
            model_grade = '우수'
        
        blind_spots = {
            '모델성능등급': model_grade,
            '주요취약점': recommendations,
            '과소예측_고위험국가': under_predicted_high_risk[['국가명', '절대오차']].to_dict('records'),
            '과대예측_저위험국가': over_predicted_low_risk[['국가명', '절대오차']].to_dict('records'),
            '개선우선순위': {
                '1순위': '고손실 국가 탐지 능력 향상',
                '2순위': '예측 안정성 개선', 
                '3순위': '저위험 국가 정확도 향상'
            }
        }
        
    except Exception as e:
        blind_spots = {'error': str(e)}
    
    return blind_spots

def calculate_risk_coverage(predicted, actual, threshold=0.8):
    """
    위험 커버리지 계산 (상위 X% 위험 국가를 얼마나 정확히 예측했는가)
    
    predicted: 예측 데이터
    actual: 실제 데이터  
    threshold: 상위 비율 (0.8 = 상위 20%)
    return: 커버리지 분석 결과
    """
    try:
        # 국가별 집계
        pred_agg = predicted.groupby('국가명')['위험지수'].mean().reset_index()
        actual_agg = actual.groupby('국가명')['보상금'].sum().reset_index()
        
        merged = pd.merge(pred_agg, actual_agg, on='국가명', how='inner')
        
        # 상위 위험 국가 식별 (실제)
        actual_threshold = merged['보상금'].quantile(threshold)
        actual_high_risk = set(merged[merged['보상금'] >= actual_threshold]['국가명'])
        
        # 상위 위험 국가 식별 (예측)
        pred_threshold = merged['위험지수'].quantile(threshold)
        pred_high_risk = set(merged[merged['위험지수'] >= pred_threshold]['국가명'])
        
        # 커버리지 계산
        true_positives = len(actual_high_risk & pred_high_risk)
        false_negatives = len(actual_high_risk - pred_high_risk)
        false_positives = len(pred_high_risk - actual_high_risk)
        
        coverage = true_positives / len(actual_high_risk) if len(actual_high_risk) > 0 else 0
        precision = true_positives / len(pred_high_risk) if len(pred_high_risk) > 0 else 0
        
        return {
            '실제_고위험국가수': len(actual_high_risk),
            '예측_고위험국가수': len(pred_high_risk),
            '정확히_예측한_국가수': true_positives,
            '커버리지': round(coverage, 4),
            '정밀도': round(precision, 4),
            '누락된_고위험국가': list(actual_high_risk - pred_high_risk),
            '잘못_예측된_국가': list(pred_high_risk - actual_high_risk)
        }
        
    except Exception as e:
        return {'error': str(e)} 