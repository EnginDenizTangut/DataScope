import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import skew
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")
st.title("📊 Veri Analiz Aracı")
st.write("Bu araç sayesinde Excel/CSV dosyanızdaki verileri kolayca inceleyebilirsiniz!")

uploaded_file = st.file_uploader("📂 Excel veya CSV dosyanızı seçin", type=["csv","xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.success(f"✅ Dosyanız başarıyla yüklendi! 📋 Toplam {data.shape[0]} satır ve {data.shape[1]} sütun bulunuyor.")
    st.subheader("📋 Verilerinizin İlk Birkaç Satırı")
    st.dataframe(data.head())

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()

    for col in data.columns:
        if col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])

    st.subheader("⚠️ Hangi Sütunlarda Boş Veri Var?")
    missing = data.isna().sum()
    missing_percent = (missing/len(data))*100
    missing_df = pd.DataFrame({"Boş Hücre Sayısı": missing, "Yüzde": missing_percent}).sort_values("Yüzde", ascending=False)

    if missing_df[missing_df["Boş Hücre Sayısı"]>0].empty:
        st.success("🎉 Harika! Verilerinizde hiç boş hücre yok.")
    else:
        st.dataframe(missing_df[missing_df["Boş Hücre Sayısı"]>0])

    if st.checkbox("Boş Verilerin Haritasını Göster"):
        fig, ax = plt.subplots(figsize=(10,4))
        sns.heatmap(data.isna(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

    st.subheader("📊 Verilerinizin Özeti")
    st.dataframe(data.describe(include='all').T)

    st.subheader("Veri Filtreleme")
    filter_col = st.selectbox("Filtrelemek istediğiniz sütunu seçin:", data.columns)
    unique_vals = data[filter_col].unique()
    selected_vals = st.multiselect("Filtrelenecek değer(ler):", unique_vals)
    if selected_vals:
        st.dataframe(data[data[filter_col].isin(selected_vals)])

    st.subheader("📊 Gruba Göre Analiz")
    group_col = st.selectbox("Gruplamak istediğiniz sütun:", data.columns)
    agg_func = st.selectbox("Uygulanacak fonksiyon:", ["mean", "sum", "count", "max", "min"])

    try:
        if agg_func == "count":
            grouped = data.groupby(group_col).count()   
        else:
            grouped = data.groupby(group_col)[numeric_cols].agg(agg_func)
        st.dataframe(grouped)
    except Exception as e:
        st.error(f"Hata: {e}")

    st.subheader("Kategorik Verileri Sayıya Çevirme")
    if len(categorical_cols) > 0:
        cat_col = st.selectbox("Dönüştürülecek sütun:", categorical_cols)
        if st.button("Label Encode"):
            le = LabelEncoder()
            data[cat_col + "_encoded"] = le.fit_transform(data[cat_col].astype(str))
            st.dataframe(data.head())
    else:
        st.info("Dönüştürülecek kategorik sütun bulunamadı.")

    st.subheader("Sütun Seçme")
    selected_columns = st.multiselect("Görmek istediğiniz sütunları seçin:", data.columns.tolist(), default=data.columns.tolist())
    st.dataframe(data[selected_columns].head())

    if numeric_cols:
        st.subheader("📈 Sayısal Verilerinizin Analizi")
        st.write("Rakamlardan oluşan sütunlarınızın detaylı incelemesi:")

        for col in numeric_cols:
            st.markdown(f"### 📊 {col}")
            col_data = data[col].dropna()
            if len(col_data) < 2:
                st.write(f"⚠️ {col} sütununda yeterli veri bulunmuyor.")
                continue

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("En Küçük Değer", f"{col_data.min():.2f}")
            with col2:
                st.metric("En Büyük Değer", f"{col_data.max():.2f}")
            with col3:
                st.metric("Ortalama", f"{col_data.mean():.2f}")
            with col4:
                st.metric("Ortanca Değer", f"{col_data.median():.2f}")

            fig, axes = plt.subplots(1,3,figsize=(18,5))

            sns.histplot(data[col].dropna(), kde=True, ax=axes[0])
            axes[0].set_title("Değerlerin Dağılımı")
            axes[0].set_ylabel("Sıklık")

            sns.boxplot(x=data[col], ax=axes[1])
            axes[1].set_title("Kutu Grafiği (Aykırı değerleri gösterir)")

            sns.scatterplot(x=data.index,y=data[col], ax=axes[2])
            axes[2].set_title("Verilerin Sıralaması")
            axes[2].set_xlabel("Satır Numarası")

            st.pyplot(fig)

            if st.checkbox("Scatter Matrix Göster", key=f"scatter_{col}"):
                from pandas.plotting import scatter_matrix
                fig = scatter_matrix(data[numeric_cols], figsize=(12,12), diagonal='kde')
                st.pyplot(plt.gcf())

            skewness = skew(col_data)
            if abs(skewness) < 0.5:
                skew_text = "Veriler dengeli dağılmış"
            elif skewness > 0.5:
                skew_text = "Veriler daha çok küçük değerler tarafında yoğunlaşmış"
            else:
                skew_text = "Veriler daha çok büyük değerler tarafında yoğunlaşmış"

            st.info(f"📈 **Dağılım Yorumu:** {skew_text}")

    st.subheader("Anamoli Analizi")
    numeric_cols = data.select_dtypes(include=['number']).columns
    col = st.selectbox("Anamoli Analizi Yapilacak Sutun: ",numeric_cols)
    method = st.radio("Method:", ["Z-score","IQR"])

    if method == "Z-Score":
        mean, std = data[col].mean(), data[col].std()
        data["Z-Score"] = (data[col] - mean) / std
        anomalies = data[np.abs(data["Z-Score"]) > 3]
        st.write("Anormal Degerler (|z| > 3):")
        st.dataframe(anomalies)
    elif method == "IQR":
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower,upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        anomalies = data[(data[col] > lower) | (data[col] > upper)]
        st.write(f"Anormal değerler (IQR yöntemi, aralık: {lower:.2f} - {upper:.2f}):")
        st.dataframe(anomalies)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.boxplot(data[col], vert=False)
    ax.set_title(f"{col} - Boxplot (Anomali Analizi)")
    st.pyplot(fig)

    if categorical_cols:
        st.subheader("📊 Metin/Kategori Verilerinizin Analizi")
        st.write("Metinlerden oluşan sütunlarınızın detaylı incelemesi:")

        for col in categorical_cols:
            st.markdown(f"### 🏷️ {col}")
            col_data = data[col].dropna()
            if len(col_data) == 0:
                st.write(f"⚠️ {col} sütununda veri bulunmuyor.")
                continue

            value_counts = data[col].value_counts()
            st.write(f"📊 Bu sütunda **{len(value_counts)}** farklı değer var")
            st.write(f"🏆 En sık görülen: **{value_counts.index[0]}** ({value_counts.iloc[0]} kez)")

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8,6))

                top_values = value_counts.head(10)
                sns.countplot(x=data[col][data[col].isin(top_values.index)], 
                            order=top_values.index, ax=ax)
                ax.set_title(f"{col} - Hangi Değer Kaç Kez Tekrarlanıyor")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(8,8))
                top_values.plot.pie(autopct="%1.1f%%", startangle=90, ax=ax2, 
                                  colors=sns.color_palette("pastel"))
                ax2.set_ylabel("")
                ax2.set_title(f"{col} - Yüzdelik Dağılım")
                st.pyplot(fig2)

            st.write("📊 **Detaylı Dağılım:**")
            percentage_df = pd.DataFrame({
                'Değer': value_counts.index,
                'Sayı': value_counts.values,
                'Yüzde': (value_counts.values / len(col_data) * 100).round(1)
            })
            st.dataframe(percentage_df.head(10))

    if len(numeric_cols) > 1:
        st.subheader("🔗 Sayısal Sütunlar Arasındaki İlişkiler")
        st.write("Hangi sayısal sütunlar birbirleriyle bağlantılı?")

        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
        ax.set_title("Sütunlar Arasındaki İlişki Haritası\n(1'e yakın = güçlü pozitif ilişki, -1'e yakın = güçlü negatif ilişki)")
        st.pyplot(fig)

        strong_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_val = corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append({
                        'Sütun 1': corr.columns[i],
                        'Sütun 2': corr.columns[j],
                        'İlişki Gücü': corr_val,
                        'Yorum': 'Güçlü pozitif ilişki' if corr_val > 0 else 'Güçlü negatif ilişki'
                    })

        if strong_corr:
            st.write("🔍 **Güçlü İlişkiler Tespit Edildi:**")
            st.dataframe(pd.DataFrame(strong_corr))
        else:
            st.info("ℹ️ Sütunlar arasında çok güçlü bir ilişki tespit edilmedi.")

    st.subheader("🔍 Belirli Bir Sütunu Detaylı İncele")
    selected_col = st.selectbox("İncelemek istediğiniz sütunu seçin", data.columns)
    if selected_col:
        st.write(f"### 📊 {selected_col} Detaylı Analizi")

        if selected_col in numeric_cols:
            col_data = data[selected_col].dropna()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ortalama", f"{col_data.mean():.2f}")
                st.metric("En Küçük", f"{col_data.min():.2f}")
            with col2:
                st.metric("Ortanca", f"{col_data.median():.2f}")
                st.metric("En Büyük", f"{col_data.max():.2f}")
            with col3:
                st.metric("Standart Sapma", f"{col_data.std():.2f}")
                st.metric("Toplam Veri", f"{len(col_data)}")

            fig, ax = plt.subplots(figsize=(8,5))
            sns.boxplot(x=data[selected_col], ax=ax)
            ax.set_title(f"{selected_col} - Kutu Grafiği")
            st.pyplot(fig)

        else:

            value_counts = data[selected_col].value_counts()
            st.write(f"**Toplam farklı değer sayısı:** {len(value_counts)}")

            fig, ax = plt.subplots(figsize=(10,6))
            value_counts.head(15).plot(kind='bar', ax=ax)
            ax.set_title(f"{selected_col} - En Sık Görülen Değerler")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            st.dataframe(value_counts.head(20))

    st.subheader("🔹 Verilerinizi Filtreleyin")
    st.write("Belirli kriterlere göre verilerinizi süzün:")

    filtered_data = data.copy()
    col_to_filter = st.selectbox("Filtrelemek istediğiniz sütunu seçin", data.columns)

    if col_to_filter in numeric_cols:
        min_val = float(data[col_to_filter].min())
        max_val = float(data[col_to_filter].max())
        selected_range = st.slider(
            f"{col_to_filter} için değer aralığı seçin", 
            min_val, max_val, (min_val, max_val)
        )
        filtered_data = filtered_data[
            (filtered_data[col_to_filter] >= selected_range[0]) & 
            (filtered_data[col_to_filter] <= selected_range[1])
        ]
        st.write(f"🔍 {selected_range[0]} ile {selected_range[1]} arasındaki veriler gösteriliyor")

    else:
        unique_vals = data[col_to_filter].dropna().unique()
        selected_vals = st.multiselect(
            f"{col_to_filter} için değerleri seçin", 
            unique_vals,
            default=list(unique_vals)[:5] if len(unique_vals) > 5 else list(unique_vals)
        )
        if selected_vals:
            filtered_data = filtered_data[filtered_data[col_to_filter].isin(selected_vals)]
            st.write(f"🔍 Seçilen değerlere sahip veriler gösteriliyor: {', '.join(map(str, selected_vals))}")

    st.write(f"📊 **Filtrelenmiş veri:** {filtered_data.shape[0]} satır, {filtered_data.shape[1]} sütun")
    st.dataframe(filtered_data)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Filtrelenmiş Veriyi Excel Olarak İndir",
            data=filtered_data.to_csv(index=False).encode('utf-8'),
            file_name='filtrelenmis_veri.csv',
            mime='text/csv'
        )

    def create_simple_pdf(df):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("📊 Veri Analiz Raporu", styles['Title']),
            Spacer(1,12)
        ]

        elements.append(Paragraph(f"Veri Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun", styles['Normal']))
        elements.append(Spacer(1,12))

        elements.append(Paragraph("📌 Temel İstatistikler", styles['Heading2']))

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

        stats_text = f"Toplam Satır: {df.shape[0]}<br/>Toplam Sütun: {df.shape[1]}<br/>"
        if numeric_cols:
            stats_text += f"Sayısal Sütun Sayısı: {len(numeric_cols)}<br/>"
        if categorical_cols:
            stats_text += f"Metin Sütun Sayısı: {len(categorical_cols)}<br/>"

        elements.append(Paragraph(stats_text, styles['Normal']))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    with col2:
        pdf_bytes = create_simple_pdf(filtered_data)
        st.download_button(
            "📄 Basit Raporu PDF Olarak İndir", 
            data=pdf_bytes, 
            file_name="veri_raporu.pdf", 
            mime="application/pdf"
        )

    if len(numeric_cols) > 1:
        st.subheader("🎯 Hangi Sütunlar Daha Önemli?")
        st.write("Bir sütunu hedef seçin, diğer sütunların bu hedefle ne kadar ilişkili olduğunu görelim:")

        target_col = st.selectbox("Hedef sütunu seçin", numeric_cols, key="target_feature")

        if target_col:

            correlations = []
            for col in numeric_cols:
                if col != target_col:
                    corr_val = data[col].corr(data[target_col])
                    if not pd.isna(corr_val):
                        correlations.append({
                            'Sütun': col,
                            'İlişki Gücü': abs(corr_val),
                            'İlişki Yönü': 'Pozitif' if corr_val > 0 else 'Negatif',
                            'Açıklama': 'Güçlü' if abs(corr_val) > 0.7 else 'Orta' if abs(corr_val) > 0.3 else 'Zayıf'
                        })

            if correlations:
                corr_df = pd.DataFrame(correlations).sort_values('İlişki Gücü', ascending=False)

                st.write(f"### 📊 {target_col} ile En Çok İlişkili Sütunlar")
                st.dataframe(corr_df)

                if len(corr_df) > 0:
                    most_correlated = corr_df.iloc[0]['Sütun']
                    fig, ax = plt.subplots(figsize=(8,6))
                    sns.scatterplot(data=data, x=most_correlated, y=target_col, ax=ax)
                    ax.set_title(f"{most_correlated} vs {target_col} İlişkisi")
                    st.pyplot(fig)

                    st.info(f"💡 **Yorum:** {most_correlated} sütunu ile {target_col} arasında "
                           f"{corr_df.iloc[0]['Açıklama'].lower()} bir {corr_df.iloc[0]['İlişki Yönü'].lower()} ilişki var.")

    if len(numeric_cols) >= 2:
        st.subheader("🤖 Basit Makine Öğrenmesi")
        st.write("Verilerinizle tahmin modelleri oluşturun!")

        ml_type = st.radio(
            "Hangi tür analiz yapmak istiyorsuniz?",
            ["Tahmin (Regression)", "Gruplandırma (Clustering)", "Sınıflandırma (Classification)"]
        )

        if ml_type == "Tahmin (Regression)":
            st.markdown("### 📈 Tahmin Modeli")
            st.write("Bir sütunu diğer sütunlara göre tahmin etmeye çalışalım!")

            target_col = st.selectbox("Tahmin edilecek sütun (hedef):", numeric_cols, key="ml_target")
            feature_cols = st.multiselect(
                "Tahmin için kullanılacak sütunlar:", 
                [col for col in numeric_cols if col != target_col],
                default=[col for col in numeric_cols if col != target_col][:3]
            )

            if len(feature_cols) > 0 and st.button("🚀 Model Oluştur ve Test Et"):
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.linear_model import LinearRegression
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.metrics import mean_squared_error, r2_score
                    import numpy as np

                    model_data = data[feature_cols + [target_col]].dropna()
                    if len(model_data) < 10:
                        st.error("⚠️ Model için yeterli veri yok (en az 10 satır gerekli)")
                    else:
                        X = model_data[feature_cols]
                        y = model_data[target_col]

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        models = {
                            "Linear Regression": LinearRegression(),
                            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
                        }

                        results = {}
                        for name, model in models.items():
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            results[name] = {
                                'model': model,
                                'mse': mse,
                                'rmse': np.sqrt(mse),
                                'r2': r2,
                                'y_pred': y_pred
                            }

                        st.success("✅ Modeller başarıyla eğitildi!")

                        col1, col2 = st.columns(2)
                        for i, (name, result) in enumerate(results.items()):
                            with col1 if i == 0 else col2:
                                st.metric(
                                    f"{name} - R² Skoru", 
                                    f"{result['r2']:.3f}",
                                    help="1'e yakın = iyi, 0'a yakın = kötü"
                                )
                                st.metric(
                                    f"{name} - RMSE", 
                                    f"{result['rmse']:.2f}",
                                    help="Düşük değer = iyi"
                                )

                        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
                        best_model = results[best_model_name]

                        st.success(f"🏆 En iyi model: **{best_model_name}** (R² = {best_model['r2']:.3f})")

                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y_test, best_model['y_pred'], alpha=0.7)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax.set_xlabel(f'Gerçek {target_col}')
                        ax.set_ylabel(f'Tahmin Edilen {target_col}')
                        ax.set_title(f'{best_model_name} - Tahmin vs Gerçek Değerler')
                        st.pyplot(fig)

                        if best_model_name == "Random Forest":
                            importance = best_model['model'].feature_importances_
                            importance_df = pd.DataFrame({
                                'Sütun': feature_cols,
                                'Önem Derecesi': importance
                            }).sort_values('Önem Derecesi', ascending=False)

                            st.write("### 📊 Hangi Sütunlar Daha Önemli?")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.barplot(data=importance_df, x='Önem Derecesi', y='Sütun', ax=ax)
                            ax.set_title('Sütunların Tahmin Gücü')
                            st.pyplot(fig)

                            st.dataframe(importance_df)

                        st.write("### 🎯 Yeni Değer Tahmini")
                        st.write("Aşağıdaki değerleri girerek tahmin yapabilirsiniz:")

                        user_input = {}
                        cols = st.columns(len(feature_cols))
                        for i, col in enumerate(feature_cols):
                            with cols[i]:
                                min_val = float(data[col].min())
                                max_val = float(data[col].max())
                                mean_val = float(data[col].mean())
                                user_input[col] = st.number_input(
                                    f"{col}:", 
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=mean_val,
                                    step=(max_val-min_val)/100
                                )

                        if st.button("🔮 Tahmin Et"):
                            input_array = np.array([[user_input[col] for col in feature_cols]])
                            prediction = best_model['model'].predict(input_array)[0]
                            st.success(f"🎯 Tahmini {target_col} değeri: **{prediction:.2f}**")

                            confidence = "Yüksek" if best_model['r2'] > 0.7 else "Orta" if best_model['r2'] > 0.3 else "Düşük"
                            st.info(f"📊 Model güvenilirliği: **{confidence}** (R² = {best_model['r2']:.3f})")

                except ImportError:
                    st.error("❌ Scikit-learn kütüphanesi yüklü değil. Lütfen 'pip install scikit-learn' komutu ile yükleyin.")
                except Exception as e:
                    st.error(f"❌ Hata oluştu: {str(e)}")

        elif ml_type == "Gruplandırma (Clustering)":
            st.markdown("### 🎯 Veri Gruplandırma")
            st.write("Verilerinizi benzerliklerine göre gruplara ayıralım!")

            cluster_cols = st.multiselect(
                "Gruplandırma için kullanılacak sütunlar:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )

            n_clusters = st.slider("Kaç grup oluşturulsun?", 2, 8, 3)

            if len(cluster_cols) >= 2 and st.button("🎯 Gruplandır"):
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler

                    cluster_data = data[cluster_cols].dropna()
                    if len(cluster_data) < n_clusters:
                        st.error(f"⚠️ Gruplandırma için yeterli veri yok (en az {n_clusters} satır gerekli)")
                    else:

                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(cluster_data)

                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(scaled_data)

                        cluster_data_with_labels = cluster_data.copy()
                        cluster_data_with_labels['Grup'] = clusters

                        st.success(f"✅ Veriler {n_clusters} gruba ayrıldı!")

                        st.write("### 📊 Grup İstatistikleri")
                        group_stats = cluster_data_with_labels.groupby('Grup').agg(['mean', 'count']).round(2)
                        st.dataframe(group_stats)

                        if len(cluster_cols) >= 2:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

                            for i in range(n_clusters):
                                cluster_points = cluster_data_with_labels[cluster_data_with_labels['Grup'] == i]
                                ax.scatter(
                                    cluster_points[cluster_cols[0]], 
                                    cluster_points[cluster_cols[1]], 
                                    c=[colors[i]], 
                                    label=f'Grup {i}',
                                    alpha=0.7
                                )

                            ax.set_xlabel(cluster_cols[0])
                            ax.set_ylabel(cluster_cols[1])
                            ax.set_title(f'Veri Gruplandırma Sonuçları')
                            ax.legend()
                            st.pyplot(fig)

                        st.write("### 🏷️ Grup Karakteristikleri")
                        for i in range(n_clusters):
                            group_data = cluster_data_with_labels[cluster_data_with_labels['Grup'] == i]
                            st.write(f"**Grup {i}** ({len(group_data)} veri noktası):")

                            characteristics = []
                            for col in cluster_cols:
                                mean_val = group_data[col].mean()
                                overall_mean = cluster_data[col].mean()
                                if mean_val > overall_mean * 1.1:
                                    characteristics.append(f"Yüksek {col}")
                                elif mean_val < overall_mean * 0.9:
                                    characteristics.append(f"Düşük {col}")

                            if characteristics:
                                st.write(f"- {', '.join(characteristics)}")
                            else:
                                st.write("- Ortalama değerlere sahip")

                        csv = cluster_data_with_labels.to_csv(index=False)
                        st.download_button(
                            "📥 Gruplandırılmış Veriyi İndir",
                            csv,
                            "gruplandırılmış_veri.csv",
                            "text/csv"
                        )

                except ImportError:
                    st.error("❌ Scikit-learn kütüphanesi yüklü değil. Lütfen 'pip install scikit-learn' komutu ile yükleyin.")
                except Exception as e:
                    st.error(f"❌ Hata oluştu: {str(e)}")

        elif ml_type == "Sınıflandırma (Classification)":
            st.markdown("### 🎯 Sınıflandırma Modeli")
            st.write("Kategorik bir sütunu diğer sütunlara göre tahmin etmeye çalışalım!")

            if len(categorical_cols) == 0:
                st.warning("⚠️ Sınıflandırma için kategorik sütun bulunamadı.")
            else:

                target_col = st.selectbox("Tahmin edilecek kategorik sütun:", categorical_cols, key="class_target")
                feature_cols = st.multiselect(
                    "Tahmin için kullanılacak sayısal sütunlar:", 
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )

                if len(feature_cols) > 0 and st.button("🚀 Sınıflandırma Modeli Oluştur"):
                    try:
                        from sklearn.model_selection import train_test_split
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.metrics import accuracy_score, classification_report
                        from sklearn.preprocessing import LabelEncoder

                        model_data = data[feature_cols + [target_col]].dropna()
                        if len(model_data) < 20:
                            st.error("⚠️ Sınıflandırma için yeterli veri yok (en az 20 satır gerekli)")
                        else:

                            le = LabelEncoder()
                            y_encoded = le.fit_transform(model_data[target_col])
                            unique_classes = len(le.classes_)

                            if unique_classes > 10:
                                st.warning("⚠️ Çok fazla kategori var. Sonuçlar karmaşık olabilir.")

                            X = model_data[feature_cols]
                            y = y_encoded

                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )

                            models = {
                                "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                            }

                            if unique_classes == 2:  
                                models["Logistic Regression"] = LogisticRegression(random_state=42, max_iter=1000)

                            best_accuracy = 0
                            best_model = None
                            best_model_name = ""

                            for name, model in models.items():
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test, y_pred)

                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_model = model
                                    best_model_name = name

                            st.success(f"✅ En iyi model: **{best_model_name}** (Doğruluk: %{best_accuracy*100:.1f})")

                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(y_test, best_model.predict(X_test))

                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                    xticklabels=le.classes_, yticklabels=le.classes_)
                            ax.set_title('Karışıklık Matrisi')
                            ax.set_xlabel('Tahmin Edilen')
                            ax.set_ylabel('Gerçek')
                            st.pyplot(fig)

                            if hasattr(best_model, 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    'Sütun': feature_cols,
                                    'Önem Derecesi': best_model.feature_importances_
                                }).sort_values('Önem Derecesi', ascending=False)

                                st.write("### 📊 Hangi Sütunlar Daha Önemli?")
                                fig, ax = plt.subplots(figsize=(8, 5))
                                sns.barplot(data=importance_df, x='Önem Derecesi', y='Sütun', ax=ax)
                                ax.set_title('Sütunların Sınıflandırma Gücü')
                                st.pyplot(fig)

                            st.write("### 🎯 Yeni Sınıflandırma Tahmini")
                            user_input = {}
                            cols = st.columns(len(feature_cols))
                            for i, col in enumerate(feature_cols):
                                with cols[i]:
                                    min_val = float(data[col].min())
                                    max_val = float(data[col].max())
                                    mean_val = float(data[col].mean())
                                    user_input[col] = st.number_input(
                                        f"{col}:", 
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=mean_val,
                                        step=(max_val-min_val)/100,
                                        key=f"class_{col}"
                                    )

                            if st.button("🔮 Sınıf Tahmini"):
                                input_array = np.array([[user_input[col] for col in feature_cols]])
                                prediction = best_model.predict(input_array)[0]
                                predicted_class = le.inverse_transform([prediction])[0]

                                if hasattr(best_model, 'predict_proba'):
                                    proba = best_model.predict_proba(input_array)[0]
                                    confidence = max(proba) * 100
                                    st.success(f"🎯 Tahmini sınıf: **{predicted_class}** (Güven: %{confidence:.1f})")
                                else:
                                    st.success(f"🎯 Tahmini sınıf: **{predicted_class}**")

                    except ImportError:
                        st.error("❌ Scikit-learn kütüphanesi yüklü değil. Lütfen 'pip install scikit-learn' komutu ile yükleyin.")
                    except Exception as e:
                        st.error(f"❌ Hata oluştu: {str(e)}")

    else:
        st.info("🤖 Makine öğrenmesi için en az 2 sayısal sütun gerekiyor.")

    st.success("✅ Analiz tamamlandı! Sorularınız varsa dosyanızın farklı bölümlerini inceleyebilirsiniz.")
