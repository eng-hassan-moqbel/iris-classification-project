#!/usr/bin/env python3
"""
الملف الرئيسي لتشغيل مشروع تصنيف زهور Iris
"""

import argparse
import sys
import os
from pathlib import Path

# إضافة مسار المشروع إلى sys.path لتجنب مشاكل الاستيراد
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    """الدالة الرئيسية لتشغيل المشروع"""
    parser = argparse.ArgumentParser(description='مشروع تصنيف زهور Iris')
    parser.add_argument('--mode', choices=['explore', 'train', 'predict', 'web', 'test'], 
                       default='train', help='وضع التشغيل')
    parser.add_argument('--features', nargs=4, type=float,
                       help='قيم الميزات للتنبؤ: sepal_length sepal_width petal_length petal_width')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'explore':
            # استكشاف البيانات
            print("🔍 استكشاف البيانات...")
            from src.data_loading import explore_dataset
            df = explore_dataset('auto')
            
        elif args.mode == 'train':
            # تدريب النماذج
            print("🔬 تدريب النماذج...")
            from src.data_loading import load_iris_data
            from src.model_training import train_models, get_best_model
            from src.model_evaluation import evaluate_model, plot_confusion_matrix, plot_feature_importance
            from src.utils import save_model
            
            X, y, feature_names, target_names = load_iris_data('auto')
            results = train_models(X, y)
            best_model, best_name, best_acc = get_best_model(results)
            
            # تقييم النموذج
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = evaluate_model(best_model, X_test, y_test, target_names)
            
            # رسم المخططات
            plot_confusion_matrix(y_test, y_pred, target_names)
            plot_feature_importance(best_model, feature_names)
            
            # حفظ النموذج
            model_path = save_model(best_model, best_name.replace(' ', '_'), best_acc)
            print(f"💾 تم حفظ النموذج في: {model_path}")
            
        elif args.mode == 'predict' and args.features:
            # التنبؤ بقيم جديدة
            print("🔮 التنبؤ بعينة جديدة...")
            from src.data_loading import load_iris_data
            from src.model_training import train_models, get_best_model
            from src.utils import predict_new_sample
            
            X, y, feature_names, target_names = load_iris_data('auto')
            results = train_models(X, y)
            best_model, best_name, best_acc = get_best_model(results)
            
            prediction = predict_new_sample(best_model, args.features, target_names)
            print(f"📊 النتائج للنموذج {best_name} (دقة: {best_acc:.2%}):")
            print(f"   التنبؤ: {prediction['prediction']}")
            print("   الاحتمالات:")
            for species, prob in prediction['probabilities'].items():
                print(f"     {species}: {prob:.2%}")
                
        elif args.mode == 'web':
            # تشغيل واجهة الويب
            print("🌐 تشغيل واجهة الويب...")
            from src.web_interface import run_web_interface
            run_web_interface()
            
        elif args.mode == 'test':
            # اختبار جميع المكونات
            print("🧪 اختبار جميع مكونات المشروع...")
            test_all_components()
            
        else:
            print("❌ يرجى تحديد وضع تشغيل صحيح")
            parser.print_help()
            
    except ImportError as e:
        print(f"❌ خطأ في الاستيراد: {e}")
        print("🔍 تأكد من:")
        print("   1. وجود جميع الملفات في مجلد src/")
        print("   2. وجود الدوال المطلوبة في الملفات")
        print("   3. تشغيل الأمر من المجلد الرئيسي للمشروع")
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        import traceback
        traceback.print_exc()

def test_all_components():
    """اختبار جميع مكونات المشروع"""
    try:
        print("🔍 اختبار تحميل البيانات...")
        from src.data_loading import load_iris_data, explore_dataset
        X, y, feature_names, target_names = load_iris_data('auto')
        df = explore_dataset('auto')
        print(f"✅ تم تحميل {len(X)} عينة بنجاح")
        
        print("🔍 اختبار تدريب النماذج...")
        from src.model_training import train_models, get_best_model
        results = train_models(X, y)
        best_model, best_name, best_acc = get_best_model(results)
        print(f"✅ أفضل نموذج: {best_name} بدقة {best_acc:.4f}")
        
        print("🔍 اختبار التقيم...")
        from src.model_evaluation import evaluate_model
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = evaluate_model(best_model, X_test, y_test, target_names)
        print("✅ تم تقييم النموذج بنجاح")
        
        print("🔍 اختبار الأدوات المساعدة...")
        from src.utils import save_model, predict_new_sample
        model_path = save_model(best_model, "test_model", best_acc)
        prediction = predict_new_sample(best_model, [5.1, 3.5, 1.4, 0.2], target_names)
        print(f"✅ التنبؤ: {prediction['prediction']}")
        
        print("🎉 جميع الاختبارات تمت بنجاح!")
        
    except Exception as e:
        print(f"❌ فشل اختبار المكونات: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()