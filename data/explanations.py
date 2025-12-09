import numpy as np

MODEL_WEIGHTS = {'rf': 0.65, 'gbm': 0.35}

def _format_val(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

def _safe(df, col):
    return df[col].iloc[-1] if col in df.columns else None

def _mean_series(series):
    try:
        return float(series.mean())
    except Exception:
        return None

def generate_detailed_explanation(signal, pred, df_feat, features, models):

    explanation = []


    if len(df_feat) > 1:
        last_close = df_feat["close"].iloc[-1]
        prev_close = df_feat["close"].iloc[-2]
        diff = last_close - prev_close
        pct = diff / prev_close * 100 if prev_close != 0 else 0.0

        if diff > 0:
            explanation.append(
                f"Цена выросла на {pct:.2f}% (с {prev_close:.2f} до {last_close:.2f})."
            )
        elif diff < 0:
            explanation.append(
                f"Цена упала на {pct:.2f}% (с {prev_close:.2f} до {last_close:.2f})."
            )
        else:
            explanation.append("Цена почти не изменилась относительно предыдущей свечи.")


    rsi = _safe(df_feat, "rsi")
    macd = _safe(df_feat, "macd")
    vwap = _safe(df_feat, "vwap")
    atr = _safe(df_feat, "atr")

    if rsi is not None:
        if rsi > 70:
            explanation.append(f"RSI={rsi:.1f} → зона перекупленности.")
        elif rsi < 30:
            explanation.append(f"RSI={rsi:.1f} → зона перепроданности.")
        else:
            explanation.append(f"RSI={rsi:.1f} → нейтральная зона.")

    if macd is not None:
        if macd > 0:
            explanation.append(f"MACD={macd:.4f} → восходящий импульс.")
        else:
            explanation.append(f"MACD={macd:.4f} → нисходящий импульс.")

    if vwap is not None:
        close = df_feat["close"].iloc[-1]
        if close > vwap:
            explanation.append(
                f"Цена ({close:.2f}) выше VWAP ({vwap:.2f}) → преобладает спрос."
            )
        else:
            explanation.append(
                f"Цена ({close:.2f}) ниже VWAP ({vwap:.2f}) → преобладает предложение."
            )

    if atr is not None:
        explanation.append(f"ATR={atr:.4f} → текущая волатильность.")


    try:
        X_row = df_feat[features].iloc[-1:].values  
        indiv_preds = {}
        for name, m in models.items():
            try:
                indiv_preds[name] = float(m.predict(X_row)[0])
            except Exception:
                indiv_preds[name] = float(m.predict(X_row.reshape(X_row.shape[1],))[0])

        preds_arr = np.array(list(indiv_preds.values()))
        mean = preds_arr.mean()
        std = preds_arr.std()
        explanation.append(f"Ансамбль моделей: средний прогноз={mean:.6f}, разброс={std:.6f}.")

        if std < (abs(mean) * 0.3 + 1e-12):
            explanation.append("Модели согласованы между собой (низкий разброс прогнозов).")
        else:
            explanation.append("Между моделями есть разногласия (высокий разброс прогнозов).")

        preds_text = ", ".join([f"{n}={v:.6f}" for n, v in indiv_preds.items()])
        explanation.append(f"По моделям: {preds_text}.")

    except Exception as e:
        explanation.append(f"Не удалось оценить согласованность ансамбля: {e}")


    feat_contrib = None
    try:
        import shap

        shap_values_weighted = None
        total_weight = 0.0
        
        X_for_shap = df_feat[features].iloc[-1:].values

        for name, m in models.items():
            weight = MODEL_WEIGHTS.get(name, 1.0)
            try:
                explainer = shap.TreeExplainer(m)
                sv = explainer.shap_values(X_for_shap)
                
                arr = np.array(sv).reshape(-1) 
                if shap_values_weighted is None:
                    shap_values_weighted = weight * arr
                else:
                    shap_values_weighted += weight * arr
                total_weight += weight
            except Exception:
                
                continue

        if shap_values_weighted is not None and total_weight > 0:
            feat_contrib = shap_values_weighted / total_weight
            
            abs_contrib = np.abs(feat_contrib)
            
            norm = abs_contrib / (abs_contrib.sum() + 1e-12)
            top_idx = np.argsort(norm)[-8:][::-1]  
            explanation.append("Вклад признаков (SHAP-псевдо, взвешенный по моделям):")
            for i in top_idx:
                feat = features[i]
                val = df_feat[feat].iloc[-1]
                contrib = feat_contrib[i]
                explanation.append(f" • {feat}: значение={_format_val(val)}; вклад={contrib:.6f}; относ.важность={norm[i]:.3f}")
        else:
            raise RuntimeError("SHAP недоступен для всех моделей или не дал результатов.")
    except Exception as e_shap:
        
        try:
            importances_weighted = None
            total_weight = 0.0
            for name, m in models.items():
                weight = MODEL_WEIGHTS.get(name, 1.0)
                if hasattr(m, "feature_importances_"):
                    imp = np.array(m.feature_importances_, dtype=float)
                    if importances_weighted is None:
                        importances_weighted = weight * imp
                    else:
                        importances_weighted += weight * imp
                    total_weight += weight

            if importances_weighted is not None and total_weight > 0:
                mean_imp = importances_weighted / total_weight
                
                norm_imp = mean_imp / (mean_imp.sum() + 1e-12)
                top_idx = np.argsort(norm_imp)[-8:][::-1]
                explanation.append("Наибольшее влияние в прогнозе оказали (по feature_importances_):")
                for i in top_idx:
                    feat = features[i]
                    val = df_feat[feat].iloc[-1]
                    imp = norm_imp[i]
                    dir_tag = ""
                    mean_val = _mean_series(df_feat[feat])
                    try:
                        if mean_val is not None and isinstance(val, (int, float)):
                            dir_tag = " (выше среднего)" if val > mean_val else " (ниже среднего)"
                    except Exception:
                        dir_tag = ""
                    explanation.append(f" • {feat}: значение={_format_val(val)}; важность={imp:.3f}{dir_tag}")
            else:
                explanation.append("Модели не предоставляют информацию о важности признаков (и SHAP недоступен).")
        except Exception as e_imp:
            explanation.append(f"Не удалось вычислить вклад признаков: {e_imp}")


    try:
        preds_list = list(indiv_preds.values())
        std = np.std(preds_list)
        mean = np.mean(preds_list)
        
        conf = max(0.0, min(1.0, (abs(mean) / (abs(mean) + std + 1e-12)) ))
        conf_pct = conf * 100
        explanation.append(f"Confidence (прибл.): {conf_pct:.1f}% (учтён разброс моделей и средний прогноз).")
    except Exception:
        explanation.append("Не удалось вычислить confidence score ансамбля.")

    if signal == "BUY":
        explanation.append(
            f"ИТОГО: модели обнаружили бычий импульс (pred={pred:.6f}), что формирует сигнал BUY."
        )
    elif signal == "SELL":
        explanation.append(
            f"ИТОГО: модели фиксируют медвежий импульс (pred={pred:.6f}), что формирует сигнал SELL."
        )
    else:
        explanation.append(
            f"ИТОГО: прогноз слабый (pred={pred:.6f}), формируется нейтральный сигнал HOLD."
        )

    return "\n".join(explanation)
