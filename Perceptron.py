import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Perceptron:
    # Konstruktor untuk menyimpan learning rate dan max epoch
    def __init__(self, alpha=0.1, epoch=10):
        self.alpha = alpha
        self.epoch = epoch

    # Fungsi menghitung nilai y_in (net input)
    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Fungsi aktivasi bipolar
    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    # Fungsi untuk visualisasi garis pemisah (Decision Boundary)
    def plot_decision_boundary(self, X, t, epoch):
        plt.figure(figsize=(8, 6))
        # Plot titik data
        plt.scatter(X[:, 0], X[:, 1], c=t.ravel(), marker='o', 
                    edgecolors='k', cmap=plt.cm.RdYlBu, s=100)
        
        # Menentukan limit grafik
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Menghitung garis pemisah: w0 + w1*x1 + w2*x2 = 0 -> x2 = -(w0 + w1*x1) / w2
        x_vals = np.linspace(x_min, x_max, 100)
        if self.w_[2] != 0:
            y_vals = -(self.w_[0] + self.w_[1] * x_vals) / self.w_[2]
            plt.plot(x_vals, y_vals, 'b', label=f'Decision Boundary (Epoch {epoch+1})')
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Decision Boundary Pada Epoch {epoch+1}")
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    # Fungsi utama untuk pelatihan (Training)
    def fit(self, X, t):
        # Inisialisasi bobot (w1, w2, ...) dan bias (w0) dengan nol
        self.w_ = np.zeros(1 + X.shape[1])
        
        # Menyimpan log hasil ke file teks
        with open("Hasil Perceptron.txt", "w") as f:
            f.write("Masalah OR dengan Perceptron\n")
            f.write("----------------------------\n")
            f.write(f"Input :\n{X}\n")
            f.write(f"Target:\n{t}\n")
            f.write(f"Bobot awal: {self.w_[1:]}\n")
            f.write(f"Bias awal : {self.w_[0]}\n")
            f.write(f"Learning rate: {self.alpha}\n")
            f.write(f"Max Epoch : {self.epoch}\n")

            for epoch in range(self.epoch):
                f.write(f"\nEpoch {epoch + 1}/{self.epoch}\n")
                f.write("-----\n")
                errors = []

                for xi, target in zip(X, t):
                    y_pred = self.predict(xi)
                    error = target[0] - y_pred
                    errors.append(error)
                    
                    # Update bobot & bias jika ada error (Delta Rule)
                    update = self.alpha * error
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    
                    f.write(f"Input: {xi}, Target: {target[0]}, Predict: {y_pred}, "
                            f"Error: {error}, Bobot: {self.w_[1:]}, Bias: {self.w_[0]}\n")

                # Hitung SSE (Sum Square Error)
                sse = sum(np.array(errors) ** 2)
                f.write(f"Sum Square Error (SSE): {sse}\n")
                
                # Visualisasi setiap epoch
                self.plot_decision_boundary(X, t, epoch)

                # Cek kondisi berhenti
                if sse == 0:
                    f.write("----------------------------\n")
                    f.write(f"Pelatihan berhenti pada epoch ke-{epoch + 1} karena SSE mencapai target (0).\n")
                    break
                elif epoch + 1 == self.epoch:
                    f.write("----------------------------\n")
                    f.write(f"Pelatihan berhenti karena max epoch ({self.epoch}) tercapai.\n")

            f.write(f"\nBobot akhir: {self.w_[1:]}\n")
            f.write(f"Bias akhir : {self.w_[0]}")