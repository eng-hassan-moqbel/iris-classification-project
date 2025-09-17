import sys
import os
import streamlit as st
import pandas as pd

# إضافة مسار المشروع إلى sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# استيراد الوحدات - استخدام استيراد مطلق
try:
    from src.data_loading import load_iris_data, check_data_files
    from src.model_training import train_models, get_best_model
    from src.utils import predict_new_sample
except ImportError as e:
    st.error(f"❌ خطأ في الاستيراد: {e}")
    # حل بديل للطوارئ
    try:
        from data_loading import load_iris_data, check_data_files
        from model_training import train_models, get_best_model
        from utils import predict_new_sample
    except ImportError as e2:
        st.error(f"❌ فشل في استيراد الوحدات: {e2}")
        raise
    

def run_web_interface():
    """تشغيل واجهة الويب باستخدام Streamlit"""
    # تهيئة الصفحة
    st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")
    st.title("🌺 نظام التعرف على زهور Iris")
    
    # التحقق من ملفات البيانات أولاً
    csv_files = check_data_files()
    
    if not csv_files:
        st.warning("""
        ⚠️ **ملاحظة**: لم يتم العثور على ملف Iris.csv في مجلد data/
        
        سيتم استخدام بيانات تدريبية افتراضية من scikit-learn.
        """)
    
    try:
        # تحميل النموذج والبيانات
        with st.spinner("جاري تحميل النموذج والبيانات..."):
            X, y, feature_names, target_names = load_iris_data('auto')
            results = train_models(X, y)
            best_model, best_name, best_acc = get_best_model(results)
        
        st.success(f"✅ تم تحميل النموذج ({best_name}) بدقة {best_acc:.2%}")
        
        # قسم إدخال البيانات
        st.header("📊 أدخل قياسات الزهرة")
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider("طول السبل (cm)", 4.0, 8.0, 5.1, 0.1)
            sepal_width = st.slider("عرض السبل (cm)", 2.0, 4.5, 3.5, 0.1)
        
        with col2:
            petal_length = st.slider("طول البتلة (cm)", 1.0, 7.0, 1.4, 0.1)
            petal_width = st.slider("عرض البتلة (cm)", 0.1, 2.5, 0.2, 0.1)
        
        # التنبؤ وعرض النتائج
        if st.button("التنبؤ بالنوع", type="primary"):
            features = [sepal_length, sepal_width, petal_length, petal_width]
            prediction = predict_new_sample(best_model, features, target_names)
            
            st.success(f"**النوع المتوقع:** {prediction['prediction']}")
            
            # عرض الاحتمالات
            st.subheader("📈 احتمالات التصنيف")
            probs_df = pd.DataFrame.from_dict(
                prediction['probabilities'], 
                orient='index',
                columns=['الاحتمالية']
            )
            st.bar_chart(probs_df)
            
            # عرض القيم المدخلة
            st.subheader("📋 القيم المدخلة")
            input_data = {
                "السمة": feature_names,
                "القيمة": features
            }
            st.table(pd.DataFrame(input_data))
        
        # قسم لعرض البيانات
        if st.checkbox("عرض البيانات المستخدمة في التدريب"):
            from src.data_loading import create_iris_dataframe
            df = create_iris_dataframe('auto')
            st.dataframe(df.head(10))
            
            # إحصائيات وصفية
            st.subheader("الإحصائيات الوصفية")
            st.write(df.describe())
            
    except Exception as e:
        st.error(f"❌ حدث خطأ: {e}")
        st.info("""
        **استكشاف الأخطاء وإصلاحها:**
        1. تأكد من وجود ملف Iris.csv في مجلد data/
        2. إذا لم يكن لديك البيانات، سيتم استخدام بيانات افتراضية
        3. تحقق من أن جميع التبعيات مثبتة بشكل صحيح
        """)

if __name__ == "__main__":
    run_web_interface()