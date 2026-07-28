import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class RaNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Inicializa os pesos e vieses de forma aleatória
        np.random.seed(42)
        self.W_in = np.random.randn(input_size, hidden_size) * 0.01
        self.b_in = np.random.randn(hidden_size) * 0.01
        self.W_out = None

    def fit_transform(self, X):
        # Passagem pela camada oculta (Randomized weights)
        H = np.tanh(np.dot(X, self.W_in) + self.b_in)
        
        # Treina os pesos de saída para reconstrução usando Ridge Regression (Pseudo-inversa regularizada)
        C = 1e-3
        # W_out = (H^T H + C * I)^-1 H^T X
        H_T_H = np.dot(H.T, H)
        I = np.eye(self.hidden_size)
        self.W_out = np.linalg.solve(H_T_H + C * I, np.dot(H.T, X))
        
        # Reconstrói os dados
        X_hat = np.dot(H, self.W_out)
        return X_hat

def main():
    img_path = 'input.jpg'
    if not os.path.exists(img_path):
        print(f"Erro: Arquivo {img_path} não encontrado.")
        return

    # 1. Ler a imagem
    img = cv2.imread(img_path)
    if img is None:
        print("Erro ao carregar a imagem.")
        return
    
    h, w, c = img.shape

    # Garantir que as dimensões sejam múltiplas de 8
    patch_size = 8
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    
    img = cv2.resize(img, (new_w, new_h))
    h, w, c = img.shape

    # 2. Fracionar em patches de 8x8
    # Reshape para (num_patches_h, patch_h, num_patches_w, patch_w, channels)
    patches = img.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
    # Transpor para ter a forma (num_patches_h, num_patches_w, patch_h, patch_w, channels)
    patches = patches.transpose(0, 2, 1, 3, 4)
    # Achatar os patches para a rede neural: (num_patches, patch_size * patch_size * c)
    num_patches = (h // patch_size) * (w // patch_size)
    patch_dim = patch_size * patch_size * c
    X = patches.reshape(num_patches, patch_dim)
    
    # Normalizar os dados para [0, 1]
    X_norm = X.astype(np.float32) / 255.0

    # 3. Rodar a rede RaNN (Autoencoder com pesos aleatórios na camada oculta)
    # Tamanho da camada oculta (reduzido para 10% do tamanho de entrada)
    hidden_size = int(patch_dim * 0.1) 
    rann = RaNN(input_size=patch_dim, hidden_size=hidden_size)
    X_hat_norm = rann.fit_transform(X_norm)
    
    # Desnormalizar
    X_hat = (X_hat_norm * 255.0).clip(0, 255).astype(np.uint8)

    # Reconstruir a imagem a partir dos patches
    patches_hat = X_hat.reshape(h // patch_size, w // patch_size, patch_size, patch_size, c)
    patches_hat = patches_hat.transpose(0, 2, 1, 3, 4)
    img_reconstructed = patches_hat.reshape(h, w, c)

    # 4. Calcular o erro de reconstrução pixel a pixel
    # Diferença absoluta entre a imagem original e a reconstruída
    error = np.abs(img.astype(np.float32) - img_reconstructed.astype(np.float32))
    # Média do erro nos canais de cor
    error_gray = np.mean(error, axis=2)
    
    # Normalizar o erro para [0, 255] para visualização
    error_normalized = cv2.normalize(error_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5. Gerar overlay azul com transparência baseada no erro
    # Transparência (alpha): 0 para acertos, 1 para erros máximos
    # Multiplicamos por 3.0 para "exagerar" a cor e torná-la bem mais visível
    alpha = (error_normalized / 255.0)
    alpha = np.clip(alpha * 3.0, 0, 1.0)[..., np.newaxis]
    
    blue_color = np.zeros_like(img)
    blue_color[:, :, 0] = 255 # BGR, canal 0 = Azul
    
    # 6. Sobrepor o azul à imagem original (Alpha blending)
    overlay = (img * (1 - alpha) + blue_color * alpha).astype(np.uint8)

    # 7. Converter o mapa de erro (escala de cinza) para BGR para concatenar
    error_map = cv2.cvtColor(error_normalized, cv2.COLOR_GRAY2BGR)

    # 8. Concatenar as 4 imagens
    # Para concatenar, criar um grid 2x2
    # O usuário pediu: original, reconstruida, mapa de erro, sobreposta
    
    # Adicionar textos nas imagens para identificação
    def add_title(image, title):
        img_copy = image.copy()
        cv2.putText(img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return img_copy

    img_titled = add_title(img, "Original")
    reconstructed_titled = add_title(img_reconstructed, "Reconstruida")
    error_map_titled = add_title(error_map, "Residuo Absoluto")
    overlay_titled = add_title(overlay, "Mapa de Complexidade Estrutural")

    # Linha 1: Original | Reconstruida
    row1 = np.hstack((img_titled, reconstructed_titled))
    # Linha 2: Residuo Absoluto | Mapa de Complexidade Estrutural
    row2 = np.hstack((error_map_titled, overlay_titled))
    
    # Imagem final
    final_result = np.vstack((row1, row2))

    # Salvar a imagem final
    cv2.imwrite('resultado.jpg', final_result)
    print("Processamento concluído. Imagem salva como 'resultado.jpg'.")

if __name__ == "__main__":
    main()
