# Car_Purchased_Decision


## Domain Proyek

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Pada zaman sekarang pembelian suatu barang sangatlah tinggi, setiap harinya banyak *customers* yang melakukan keputusan untuk membeli produk atau tidak[2]. *E-commerce* maupun penjualan barang-barang mewah seperti mobil, motor, perhiasan, dan lain-lain sangat mudah ditemui. 
Dengan adanya teknologi, setiap manusia dapat melihat penjualan-penjualan tersebut. Tidak hanya itu dengan adanya teknologi juga dapat menguntungkan para penjual
dikarenakan penjualan dapat dengan mudah dilakukan dengan *gadget*. Para pengusaha yang melakukan penjualan pun perlu mengetahui pembeli mereka apakah mereka akan membeli
barang tersebut atau tidak. Dengan mengetahui pembeli-pembeli mereka, pengusaha akan dengan mudah untuk menentukkan target market mereka dan tentu saja ini akan menambah
penghasilan dari suatu perusahaan. Sebagai salah satu faktor yang mempengaruhi perilaku pembelian konsumen, gaya *decision-making* sangat penting untuk
memahami perilaku belanja konsumen dan untuk mengembangkan strategi pemasaran yang sukses[1]\.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oleh karena itu, dengan adanya *machine learning* dapat membantu pengusaha untuk mengenali lebih dalam *target market* mereka dan menambah penghasilan suatu perusahaan.
Pada studi kasus ini, contoh pengenalan pembeli yang diambil adalah penjualan mobil untuk menentukkan keputusan penjualan terhadapa calon pembeli\.

**Alasan penyelesaian masalah**:
- Masalah ini perlu diselesaikan karena jika tidak maka pengusaha kesulitan untuk mengetahui target market mereka dan akhirnya hanya akan membuang uang untuk melakukan promosi ke target market yang salah\.

## *Business Understanding*

### Problem Statements

Menjelaskan pernyataan masalah latar belakang\:
- Bagaimana cara untuk mengenali *customer* apakah mereka ingin membeli atau tidak?\.
- Bagaimana cara untuk memahami pasar agar perusahaan dapat mengetahui sasaran pasar yang baik?\.

### *Goals*

Menjelaskan tujuan dari pernyataan masalah\:
- Menganalisis data pelanggan sebagai referensi untuk calon pembeli apakah calon pembeli akan membeli atau tidak\.
- Menggunakan machine learning untuk membantu memahami pasar dengan desicion making analysis dari data pelanggan\.

## *Data Understanding*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Data yang digunakan merupakn data penjualan mobil yang diambil dari kaggle Dataset memiliki jumlah baris yaitu 1000 baris dan 5 kolom. Isi dari dataset ini adalah detail mengenai 1000 customers yang memiliki kecenderungan untuk membeli mobil, berdasarkan dari penghasilan tahunan mereka\.

[Cars - Purchased Decision Dataset](https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset)\.

### Variabel-variabel pada *Cars - Purchased Decision* dataset adalah sebagai berikut\:
- Users ID : merupakan id customers untuk membedakan tiap customers\.
- Age : merupakan usia dari tiap customers\.
- AnnualSalaries: merupakan penghasilan tahunan dari tiap customers\.
- Purchased: merupakan target kolom sebagai penentu apakah customer akan membeli mobil (1) atau tidak (0)\.

**Penjelasan lengkap mengenai *Exploratory Data Analysis* untuk memahami data**\:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Pada project ini dilakukan *exploratory* pada dataset untuk mengetahui informasi dataset lebih lanjut, adapun tahapan *exploratory* dataset sebagai berikut\:

- ***data.info()***

![image](https://user-images.githubusercontent.com/91602612/183915900-f6da52d6-2bc6-4c2d-9c8e-62ea4b8b9b8e.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Untuk melihat informasi semua *features* yang ada di dataset. Pada tahap ini informasi yang diberikan adalah berupa informasi umum yang ada di dataset, seperti *range index* yaitu dari 0 - 999, jumlah baris yaitu 1000, jumlah kolom yaitu 5 (*User ID, Gender, Age, AnnualSalary, Purchased*), *datatypes* yaitu 4 kolom *dtype*-nya adalah int64, dan 1 kolom *dtype*-nya adalah *object* dan penggunaan memori dari datasset yaitu 39,2+ mb\.

- ***data.describe()***

![image](https://user-images.githubusercontent.com/91602612/183916121-b9a8f5cc-6b3a-4f44-a0a1-ccfb5b2204eb.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Untuk melihat informasi semua *features* secara statistik seperti *mean*, *std* tiap kolom, dll\.

- ***data.isnull().sum()*** 

![image](https://user-images.githubusercontent.com/91602612/183917694-360d8d64-0a5d-4a29-b318-cff38d3962a3.png)
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Untuk melihat jumlah *missing value* yang ada pada tiap *features*. Dan bisa dilihat bahwa sudah tidak ada *missing value* pada setiap fitur\.

- ***plot heatmap correlation***

![image](https://user-images.githubusercontent.com/91602612/183862246-dbe96bc7-2d09-42e5-8dd0-83ff76153c3a.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Gambar di atas merupakan *plot* menggunakan *heat map* untuk melihat korelasi dari tiap fitur yang ada di dataset. pada *matrix correlation* di atas bisa dilihat bahwa fitur *User ID* dan *Gender* memiliki korelasi yang sangat rendah terhadap tiap fitur yang ada di dataset dan memiliki korelasi yang rendah juga terhadap kolom *target / label (purchases)*, oleh karena itu kita perlu untuk menghilangkan kedua fitur tersebut agar akurasi yang akan dihasilkan nantinya menjadi lebih baik\. 

- ***data['Purchased'].value_counts()*** 
 
 ![image](https://user-images.githubusercontent.com/91602612/183919366-ae921a08-c93c-4edb-9d8d-837b506319af.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Untuk melihat berapa banyak jumlah *customers* yang membeli dan yang tidak membeli, di sini *customer* yang membeli di representasikan dengan angka 1 yang berjumlah 402 (402 *customer* yang membeli), sedangkan untuk *customer* yang tidak membeli di representrasikan dengan angka 0 yang berjumlah 598 (598 *customer* dari 1000 yang tidak membeli)\.

- **max() dan min() pada age dan annual salaries**:
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Untuk melihat berapa usia tertinggi dan terendah (usia tertinggi adalah 63, dan usia terendah adalah 18), dan penghasilan tertinggi dan terendah (penghasilan tertinggi adalah 152500 dan penghasilan terendah adalah 15000)\.

- ***histogram plot***

![image](https://user-images.githubusercontent.com/91602612/183920963-4af2b7f0-6582-4b23-9f77-03c6c1bef434.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;untuk memvisualisasikan fitur yaitu *age* dan *annual salary*. Pada histogram di atas, dapat diketahui bahwa *customer* paling banyak ada di umur 38-40, dan juga *customer* dengan penghasilan terbanyak ada pada kisaran 7000-7500\.  

- ***box plot***

![image](https://user-images.githubusercontent.com/91602612/183921945-5cad4ea4-c163-4913-be3b-4f4b6d9384d7.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Untuk memvisualisasikan *features* guna mencari *outliers*, akan tetapi setelah melihat gambar di atas diketahui bahwa tidak ada data yang *outliers*\.

## *Data Preparation*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Data preparation* yang dilakukan pada project ini adalah\: 
- Dengan menggunakan *one hot encoding* untuk mengubah data categorical menjadi data numerical\. 
- Melakukan standarisasi data\. 
- Melakukan *train test split data*\.

**Detail persiapan data**: 
- Pada data preparation yang pertama dilakukan adalah dengan membuat variabel x (independent) yaitu kolom *Age* dan *AnnualSalary* dan variable y (dependent) yaitu kolom *Purchased*: dengan membuat variabel x dan mengisinya dengan *data[['Age', 'AnnualSalary']].values* dan membuat variable y dengan isinya yaitu *data['Purchased]*\.
- melakukan nomralisasi dengan StandardScaler() terhadap data X dikarenakan data X memiliki nilai yang tidak seimbang dalam skala oleh karena itu diperlukan normalisasi agar skala dari nilainya yaitu 0-1\.
- Melakukan train test split data agar dapat membedakan data train yang akan digunakan untuk training model dan data testing yang digunakan untuk menguji model dengan data baru. Menggunakan *library scikit-learn* yaitu *train_test_split* untuk mempermudah melakukannya dengan parameter *test_size* yaitu 0.2 yang berarti memisahkan data *test* sehingga berjumlah 20% dari semua data yang ada dan data *train* menjadi 80%, tidak hanya itu digunakan juga parameter *random_state=10* untuk mengacak data\.

## *Modeling*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Untuk modeling, algoritma yang digunakan adalah *Random Forest Classifier* dikarenakan algoritma ini merupakan salah satu algoritma *supervised learning* dan termasuk dalam kasus klasifikasi. Parameter yang digunakan untuk model ini adalah n_estimators = 400 dan citerion = gini. parameter ini digunakan karena hasil parameter terbaik berdasarkan *grid seacrh*\.

## *Evaluation*
Pada tahap evaluasi, project ini menggunakan:
- mean squared error: 12%
- accuracy score: 87%
- confussion matrix: (tp: 114, tn: 12, fp: 13, fn: 61)
- Classification report: 

![image](https://user-images.githubusercontent.com/91602612/184045644-50e35b30-f00d-4e04-b5e7-77babad35d38.png)

## *Conclusion*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Decision-making* yang dilakukan untuk melakukan pembelian atau tidak terhadap suatu produk sering dialami oleh *customer* tentunya tidak semua *customer* dapat membeli atau memiliki keinginan untuk membeli produk tersebut. Hal itu dapat membuat perusahan kesulitan untuk mengerti calon pembeli yang pas untuk produknya. Akan tetapi dengan adanya *machine learning* hal itu dapat diatasi, dengan menggunakan algoritma *Random Forest Classsifier* untuk mengklasifikasi *customer* yang ingin membeli dan yang tidak membeli. Dengan itu perusahaan dapat mengerti tipikal pembeli yang sesuai dengan produk dan meningkatkan pemasaran dan mengerti *target* pasar yang ingin dicapai. Studi kasus ini mengangkat persoalan *desicion-making* pada penjualan mobil, dengan menggunakan algoritma *machine learning*, kita bisa tahu mana saja pembeli yang akan melakukan pembelian mobil, untuk akurasi dari modelnya sendiri ialah 87% dan *mean squared error* hanya 12%\.

## *References*
[1]Hamedani, Seyed E.A.H, *"A Review of Consumer Decision-Making Styles: Existing Styles and Proposed Additional Styles"* vol.7, 2017\.

[2]Stubeid, Saavi, and Arandjelovic, Ognjen, *"Machine Learning Based Prediction of Consumer Purchasing Decisions: The Evidence and Its Significance"* 2018\.



**

