# Gerekli kütüphaneleri içe aktarıyoruz

# Veri işleme ve analiz için pandas
import pandas as pd 
# pandas, veri analizi ve manipülasyonu için kullanılan güçlü bir kütüphanedir.
# Veri setlerini DataFrame formatında işler ve üzerinde kolayca işlemler yapılmasını sağlar.

# Sayısal işlemler için numpy
import numpy as np 
# numpy, büyük veri setleri üzerinde yüksek performanslı matematiksel işlemler yapmamızı sağlar.
# Diziler (array) ile çalışmak için kullanılır ve vektörel işlemleri kolaylaştırır.

# Grafik çizimi için matplotlib
import matplotlib.pyplot as plt 
# matplotlib, grafik çizimi için kullanılan popüler bir kütüphanedir.
# Veri analizinin görselleştirilmesini sağlar, histogram, çizgi grafiği gibi birçok grafik türü oluşturabiliriz.

# Gelişmiş görselleştirme için seaborn
import seaborn as sns 
# seaborn, matplotlib'in üzerine inşa edilmiş, daha gelişmiş ve estetik grafikler oluşturmaya yardımcı olan bir kütüphanedir.
# Veri setlerinin dağılımını ve ilişkilerini daha kolay analiz etmemizi sağlar.

# Veri standardizasyonu için StandardScaler
from sklearn.preprocessing import StandardScaler 
# StandardScaler, veriyi standart normal dağılıma dönüştürmek için kullanılır.
# Verinin ortalamasını 0'a, standart sapmasını 1'e getirerek ölçeklendirme yapar.
# Bu, özellikle mesafe tabanlı algoritmalar (KNN, SVM) ve Gradient Descent tabanlı algoritmalar için büyük önem taşır.
# Ölçeklendirilmemiş veriler, modelin yanlış öğrenmesine veya bazı değişkenlerin diğerlerine baskın olmasına neden olabilir.

# Veri setini eğitim ve test kümelerine ayırmak için train_test_split
# Model değerlendirmesi için KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split , KFold, cross_val_score, GridSearchCV 
# train_test_split: Veriyi eğitim ve test kümelerine ayırarak modelin performansını ölçmemizi sağlar.
# KFold: K-katlı çapraz doğrulama yöntemiyle modeli farklı veri bölümlerinde test etmemize olanak tanır.
# cross_val_score: Modelin çapraz doğrulama ile değerlendirilmesini sağlar.
# GridSearchCV: Hiperparametre optimizasyonu yaparak en iyi model parametrelerini belirler.

# Sınıflandırma modellerinin performansını değerlendirmek için classification_report ve confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix 
# classification_report: Modelin doğruluk, hassasiyet, f1 skoru gibi performans metriklerini detaylı şekilde verir.
# confusion_matrix: Modelin tahmin ettiği ve gerçek değerler arasındaki ilişkiyi gösteren hata matrisi oluşturur.

# Lojistik regresyon modeli
from sklearn.linear_model import LogisticRegression 
# LogisticRegression: İkili sınıflandırma problemleri için kullanılan istatistiksel bir modeldir.
# Olasılık temelli çalışarak verileri sınıflandırır ve sigmoid fonksiyonu ile çıktı üretir.

# Karar ağacı sınıflandırıcısı
from sklearn.tree import DecisionTreeClassifier 
# DecisionTreeClassifier: Veri seti içindeki özelliklere dayalı kararlar vererek sınıflandırma yapan bir algoritmadır.
# Ağaç yapısı oluşturarak verileri dallara ayırır ve tahminler yapar.

# K-En Yakın Komşu (KNN) algoritması
from sklearn.neighbors import KNeighborsClassifier 
# KNeighborsClassifier: Verinin en yakın K komşusuna bakarak sınıflandırma yapar.
# Mesafe tabanlı bir algoritma olduğu için ölçeklendirme gereklidir.

# Naive Bayes sınıflandırıcısı
from sklearn.naive_bayes import GaussianNB 
# GaussianNB: Olasılık tabanlı bir algoritmadır ve Bayes teoremini kullanarak tahmin yapar.
# Özellikle metin madenciliği ve spam tespitinde kullanılır.

# Destek Vektör Makineleri (SVM) sınıflandırıcısı
from sklearn.svm import SVC 
# SVC: Destek vektör makineleri, verileri sınıflar arasında en iyi şekilde ayıran bir hiper düzlem bulmaya çalışır.
# Küçük veri setlerinde yüksek performans gösterir.

# Topluluk (ensemble) öğrenme yöntemleri
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier 
# AdaBoostClassifier: Zayıf öğrenicileri birleştirerek güçlü bir sınıflandırıcı oluşturur.
# GradientBoostingClassifier: Hata oranını azaltarak daha iyi tahminler yapmak için gradyan artırma yöntemini kullanır.
# RandomForestClassifier: Birden fazla karar ağacı oluşturarak daha kararlı ve güçlü bir model üretir.

# Uyarıları görmezden gelmek için
import warnings 
warnings.filterwarnings("ignore") 
# Çalışma sırasında oluşabilecek gereksiz uyarıları gizleyerek kodun daha temiz çalışmasını sağlar.



df = pd.read_csv("diabetes.csv") 
 # pandas kütüphanesinin 'read_csv' fonksiyonu ile "diabetes.csv" dosyasındaki verileri okur ve df değişkenine yükler. 
 # #Bu, verileri bir DataFrame'e dönüştürür. DataFrame, tablo şeklinde düzenlenmiş verilerin saklanmasını sağlar.
df_name = df.columns  # df veri çerçevesinin (DataFrame) sütun adlarını alır. 'df.columns' komutu, 
#veri çerçevesindeki tüm sütun isimlerini döndürür. Bu, hangi verilerin hangi sütunlarda olduğunu görmek için kullanılır.
df.info()  # dfsayısını gösterir. veri çerçevesi hakkında genel bir bilgi verir. Bu komut, veri setindeki satır sayısını, sütun sayısını, her sütunun 
#veri tiplerini (örneğin int64, float64) ve eksik değerlerin 
df.describe()  # df veri çerçevesinin sayısal sütunlarının temel istatistiksel özetini verir. Özet, ortalama, standart sapma, minimum ve maksimum değerler ile 
#çeyrek dilimlere (Q1, Q2, Q3) ilişkin bilgileri içerir.
describe = df.describe()  # 'df.describe()' fonksiyonunun çıktısını 'describe' değişkenine atar. Bu, daha sonra bu özet istatistikleri üzerinde işlem yapabilmek için
#kullanılır.
sns.pairplot(df, hue="Outcome")  # Seaborn kütüphanesinin 'pairplot' fonksiyonunu kullanarak, veri çerçevesindeki
#tüm sayısal sütunlar arasındaki ilişkileri görselleştirir. 'hue="Outcome"' parametresi, "Outcome" sütununa göre verileri renklendirir 
#ve diyabetli ve diyabetsiz kişiler arasındaki farkları görselleştirir.
plt.show()  # matplotlib kütüphanesinin 'show()' fonksiyonunu çağırarak, daha önce oluşturulan grafikleri ekranda gösterir. 
#Bu komut, görselleştirilen verilerin ekranda görüntülenmesini sağlar.

# plot_correlation_heatmap fonksiyonu, verilen veri çerçevesi (df) üzerinde korelasyon ısı haritası (heatmap) çizer
def plot_correlation_heatmap(df):
    
    # Veri çerçevesindeki sayısal sütunlar arasındaki korelasyon matrisini hesapla
    corr_matrix = df.corr()
    
    # Grafik boyutunu ayarla: genişlik 10, yükseklik 8
    plt.figure(figsize=(10, 8))
    
    # Korelasyon matrisini ısı haritası olarak çiz
    # annot=True: Hücrelerde korelasyon değerlerinin yazdırılmasını sağlar
    # cmap="coolwarm": Soğuk ve sıcak renk paleti kullanılarak korelasyon değerleri renklerle gösterilir
    # fmt=".2f": Sayılar iki ondalık basamağa yuvarlanarak gösterilecektir
    # linewidths=0.5: Hücreler arasındaki çizgilerin kalınlığı 0.5 olacak şekilde ayarlanır
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    
    # Grafiğe başlık ekle
    plt.title("Correlation of Features")
    
    # Grafiği göster
    plt.show()

# Verilen veri çerçevesi (df) ile fonksiyonu çağır
plot_correlation_heatmap(df)

import pandas as pd

def detect_outliers_iqr(df):
    outlier_indices = []  # Aykırı değerlerin indekslerini saklamak için boş bir liste oluşturuyoruz.
    outliers_df = pd.DataFrame()  # Aykırı değerleri içerecek boş bir DataFrame oluşturuyoruz.

    # Yalnızca sayısal (float64 ve int64) sütunları seçiyoruz.
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = df[col].quantile(0.23)  # 1. çeyrek (Q1)
        Q3 = df[col].quantile(0.75)  # 3. çeyrek (Q3)
        IQR = Q3 - Q1  # Çeyrekler arası açıklık

        lower_bound = Q1 - 2.5 * IQR  # Alt sınır
        upper_bound = Q3 + 2.5 * IQR  # Üst sınır

        # Aykırı değerleri seç
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        outlier_indices.extend(outliers_in_col.index)  # İndeksleri ekle
        outliers_df = pd.concat([outliers_df, outliers_in_col], axis=0)  # Aykırıları birleştir

    # Tekrar eden indeksleri kaldır
    outlier_indices = list(set(outlier_indices))
    outliers_df = outliers_df.drop_duplicates()

    return outliers_df, outlier_indices  


# Örnek bir DataFrame'in aykırı değerlerini tespit etmek için fonksiyonu çağırıyoruz.
outliers_df, outlier_indices = detect_outliers_iqr(df)

# Kaç tane aykırı değerin silindiğini ekrana yazdırıyoruz.
print(f"Toplam {len(outlier_indices)} aykırı değer silindi.")

# Aykırı değerleri veri setinden kaldırıyoruz ve indeksleri sıfırlıyoruz.
df_cleaned = df.drop(outlier_indices).reset_index(drop=True)


# Train test split
x = df_cleaned.drop(["Outcome"], axis=1)
y = df_cleaned["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Standartizasyon
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)  # fit_transform kullanıldı
x_test_scaled = scaler.transform(x_test)  # transform kullanıldı


# Model listesi oluşturan fonksiyon
def getBaseModel():
    """Farklı makine öğrenmesi modellerini bir liste olarak döndürür."""
    baseModels = []
    baseModels.append(("LR", LogisticRegression()))  # Lojistik Regresyon
    baseModels.append(("DT", DecisionTreeClassifier()))  # Karar Ağacı
    baseModels.append(("KNN", KNeighborsClassifier()))  # K-En Yakın Komşu (Hata düzeltildi: () eklendi)
    baseModels.append(("NB", GaussianNB()))  # Naive Bayes
    baseModels.append(("SVM", SVC()))  # Destek Vektör Makineleri
    baseModels.append(("AdoB", AdaBoostClassifier()))  # AdaBoost
    baseModels.append(("GBM", GradientBoostingClassifier()))  # Gradient Boosting
    baseModels.append(("RF", RandomForestClassifier()))  # Rastgele Orman

    return baseModels  # Model listesini döndürür

# Modelleri eğitme ve çapraz doğrulama ile değerlendirme fonksiyonu
def baseModelsTraining(x_train, y_train, models):
    """
    Verilen eğitim verileri ile modelleri eğitir ve çapraz doğrulama uygular.
    10 katlı çapraz doğrulama ile modellerin doğruluğunu hesaplar.
    """
    results = []  # Sonuçları saklamak için liste
    names = []  # Model isimlerini saklamak için liste

    for name, model in models:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10 katlı çapraz doğrulama (shuffle=True eklendi)
        cv_results = cross_val_score(model, x_train, y_train, cv=kf, scoring="accuracy")  # Modelin doğruluğunu hesapla
        results.append(cv_results)  # Sonuçları listeye ekle
        names.append(name)  # Model ismini listeye ekle
        print(f"{name}: accuracy: {cv_results.mean():.4f}, std: {cv_results.std():.4f}")  # Ortalama ve standart sapmayı yazdır

    return names, results  # Model isimleri ve doğruluk sonuçlarını döndür

# Boxplot ile model başarılarını görselleştirme fonksiyonu
def plot_Box(names, results):
    """
    Modellerin doğruluk değerlerini kutu grafiği (boxplot) ile görselleştirir.
    """
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})  # Model sonuçlarını DataFrame'e çevir
    plt.figure(figsize=(12, 8))  # Grafik boyutunu belirle
    sns.boxplot(data=df)  # Seaborn ile kutu grafiği çiz
    plt.title("Model Accuracy")  # Başlık ekle
    plt.ylabel("Accuracy")
    plt.show()  # Grafiği göster

# Model eğitim ve değerlendirme işlemi
models = getBaseModel()  # Model listesini al
names, results = baseModelsTraining(x_train, y_train, models)  # Modelleri eğit ve değerlendir

# Sonuçları görselleştir
plot_Box(names, results)


#hyperparameter tuning
# Hiperparametre ayarlaması için kullanılacak parametreleri belirliyoruz

param_grid = { # Bu satır, bir Python sözlüğü (dictionary) oluşturuyor. 
    #Bu sözlük, makine öğrenmesi modelini optimize etmek için kullanılacak 
   # hiperparametrelerin isimlerini ve bu parametreler için denenecek olası değerleri içeriyor.
    "criterion": ["gini", "entropy"],  
    #🔹 criterion: Karar ağacı modelinin hangi ölçütü kullanarak bölünme (split) yapacağını belirleyen parametredir.
#"gini": Gini indeksini kullanarak düğüm bölme işlemini yapar. Düşük Gini değeri daha saf (homojen) veri kümeleri oluşturur.
#"entropy": Entropi ölçütünü kullanarak düğüm bölme işlemini yapar. Bilgi kazancına dayanarak bölmeyi belirler.
#🔹 Bu parametre, modelin neye göre dallanacağını kontrol eder.
    "max_depth": [10, 20, 30, 40, 50],  
     #max_depth: Karar ağacının maksimum derinliğini belirleyen parametredir.
#Daha düşük değerler (örneğin, 10 veya 20) modeli basitleştirerek aşırı öğrenmeyi (overfitting) engelleyebilir.
#Daha yüksek değerler (örneğin, 40 veya 50) ağacı daha karmaşık hale getirerek veriye daha iyi uyum sağlayabilir ama aşırı öğrenme riski taşır.
#🔹 Bu parametre, ağacın ne kadar derine inebileceğini sınırlar.
    "min_samples_split": [2, 5, 10],  
     #min_samples_split: Bir düğümün dallanabilmesi için en az kaç örnek (sample) içermesi gerektiğini belirler.
#2: Eğer düğümde en az 2 örnek varsa, bölünebilir. (Daha küçük ve derin ağaçlar oluşur.)
#5: Bir düğümde en az 5 örnek olduğunda bölünebilir.
#10: En az 10 örnek varsa bölünebilir. (Daha büyük ve daha az dallanmış bir ağaç oluşur.)
#🔹 Bu parametre, ağacın aşırı detaycı olmasını engellemek için kullanılır.
    "min_samples_leaf": [1, 2, 4] 
    #🔹 min_samples_leaf: Bir yaprak düğümünde (leaf node) bulunması gereken en az örnek sayısını belirler.
#1: Yaprak düğümlerinde tek bir örnek olabilir. (Ağaç daha detaylı öğrenir.)
#2: Yaprak düğümlerinde en az 2 örnek olmalıdır.
#4: Yaprak düğümlerinde en az 4 örnek olmalıdır. (Ağaç daha az dallanır, genelleme gücü artabilir.)
#🔹 Bu parametre, aşırı öğrenmeyi önlemek için yaprak düğümlerin minimum büyüklüğünü belirler.
}


# Karar ağacı modelini oluşturuyoruz
dt = DecisionTreeClassifier()

# Grid Search CV (GridSearchCV), en iyi parametre kombinasyonunu belirlemek için çapraz doğrulama uygular
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy")
#GridSearchCV, verilen parametrelerin tüm kombinasyonlarını deneyerek en iyi sonucu veren parametreleri seçmeye çalışır.
#Veri seti 5 eşit parçaya bölünür. Model 5 kez eğitilip test edilir ve her seferinde farklı bir kısım test için kullanılır.
#Accuracy (Doğruluk), doğru sınıflandırılan örneklerin toplam örneğe oranıdır.
# Modeli eğitim verileri ile eğitiyoruz


grid_search.fit(x_train, y_train)

# En iyi parametreleri ekrana yazdırıyoruz
print("En iyi parametreler:", grid_search.best_params_)

# En iyi bulunan modeli alıyoruz
best_dt_model = grid_search.best_estimator_

# Test verileri ile tahmin yapıyoruz
y_pred = best_dt_model.predict(x_test)
#🔹 y_pred (Tahmin Edilen Etiketler - Predicted Labels):
# Modelin x_test verileri için tahmin ettiği sonuçları içerir.

# Confusion Matrix'i ekrana yazdırıyoruz
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Sınıflandırma raporunu ekrana yazdırıyoruz
print("Classification Report")
print(classification_report(y_test, y_pred))

# Model testing with real data


# Burada aşağıdaki adımlar gerçekleştirilecek:
# 1. Veriyi içe aktarma ve Keşifsel Veri Analizi (EDA)
# 2. Aykırı değer (outlier) tespiti
# 3. Veri setinin eğitim ve test kümelerine ayrılması
# 4. Verinin standartizasyonu
# 5. Model eğitimi ve değerlendirmesi
# 6. Hiperparametre ayarı (hyperparameter tuning)
# 7. Modelin gerçek veri ile test edilmesi

