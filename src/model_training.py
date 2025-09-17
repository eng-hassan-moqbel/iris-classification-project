from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    تحضير البيانات للتدريب
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_models(X, y):
    """
    تدريب نماذج مختلفة ومقارنة أدائها
    """
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results[name] = {
            'model': model,
            'accuracy': accuracy
        }
        print(f"{name}: {accuracy:.4f}")
    
    return results

def get_best_model(results):
    """
    اختيار أفضل نموذج بناءً على الدقة
    """
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"🎯 أفضل نموذج: {best_model_name} بدقة {best_accuracy:.4f}")
    return best_model, best_model_name, best_accuracy

if __name__ == "__main__":
    from data_loading import load_iris_data
    print("🔬 بدء تدريب النماذج...")
    X, y, _, _ = load_iris_data('csv')
    results = train_models(X, y)
    best_model, best_name, best_acc = get_best_model(results)