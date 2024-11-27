import pandas as pd

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    money_loss = X_test[(y_test != y_pred) & ((y_test.str.contains("_Churn")) | (y_test.str.contains("Yes")))].TotalCharges.sum() 
    risk_factor = X_test[(y_test != y_pred) & ((y_test.str.contains("_Churn")) | (y_test.str.contains("Yes")))].TotalCharges.sum() / X_test[(y_test.str.contains("_Churn")) | (y_test.str.contains("Yes"))].TotalCharges.sum()
    # false_count = len(y_pred[y_test != y_pred])
    false_count = X_test[y_test != y_pred].TotalCharges.count()
    false_neg = X_test[(y_test != y_pred) & ((y_test.str.contains("_Churn")) | (y_test.str.contains("Yes")))].TotalCharges.count()
    return money_loss,risk_factor,false_count,false_neg
    
def evaluate_models(models,X_test,y_test):
    results = {}
    m,m_loss, r_factor ,f_count,f_neg = [],[],[],[],[]
    for model,name in models:
        money_loss,risk_factor,false_count,false_neg = evaluate_model(model,X_test,y_test)
        m_loss.append(money_loss)
        r_factor.append(risk_factor)
        f_count.append(false_count)
        m.append(str(name))
        f_neg.append(false_neg)
    results["Models"] = m
    results["Potential Risk in Currency"] = m_loss
    results["Risk Factor"] = r_factor
    results["Number of wrong predictions"] = f_count
    results["False Negatives"] = f_neg
    return pd.DataFrame(results)
    
    