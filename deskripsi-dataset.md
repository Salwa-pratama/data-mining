# 📊 Exploratory Data Analysis - Loan Sanction Dataset

## 🧠 Deskripsi Dataset

Dataset ini digunakan untuk memprediksi apakah suatu pengajuan pinjaman akan disetujui atau tidak berdasarkan berbagai faktor seperti pendapatan, status pernikahan, dan riwayat kredit.

Dataset terdiri dari **614 data (baris)** dan beberapa fitur numerik serta kategorikal.

---

## 📂 Struktur Dataset

| No | Kolom               | Non-Null Count | Tipe Data | Deskripsi |
|----|--------------------|---------------|----------|----------|
| 0  | Loan_ID            | 614           | string   | ID unik untuk setiap pengajuan |
| 1  | Gender             | 601           | string   | Jenis kelamin pemohon |
| 2  | Married            | 611           | string   | Status pernikahan |
| 3  | Dependents         | 599           | string   | Jumlah tanggungan (keluarga) |
| 4  | Education          | 614           | string   | Tingkat pendidikan |
| 5  | Self_Employed      | 582           | string   | Status pekerjaan (wirausaha atau tidak) |
| 6  | ApplicantIncome    | 614           | int64    | Pendapatan pemohon |
| 7  | CoapplicantIncome  | 614           | float64  | Pendapatan co-pemohon |
| 8  | LoanAmount         | 592           | float64  | Jumlah pinjaman yang diajukan |
| 9  | Loan_Amount_Term   | 600           | float64  | Jangka waktu pinjaman |
| 10 | Credit_History     | 564           | float64  | Riwayat kredit (0 = buruk, 1 = baik) |
| 11 | Property_Area      | 614           | string   | Lokasi properti |
| 12 | Loan_Status        | 614           | string   | Status persetujuan pinjaman (Target) |

---

## ⚠️ Missing Values

Beberapa kolom memiliki data yang hilang:

- Gender → 13 missing
- Married → 3 missing
- Dependents → 15 missing
- Self_Employed → 32 missing
- LoanAmount → 22 missing
- Loan_Amount_Term → 14 missing
- Credit_History → 50 missing

👉 Perlu dilakukan handling missing values sebelum modeling.

---


