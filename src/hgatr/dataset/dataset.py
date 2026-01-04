import numpy as np
import torch
import random
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

def reduce_data_bands(data, n_bands, ica = "False"):
    # Rimodella i dati per l'applicazione della PCA (flattening spaziale)
    data_reshaped = data.reshape(-1, data.shape[-1])  # shape: (145*145, 200)

    # Standardizza i dati
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)  # shape: (145*145, 200)

    # Applica PCA per ridurre i canali a 48
    n_components_pca = n_bands
    pca = PCA(n_components=n_components_pca)
    data_reduced = pca.fit_transform(data_scaled)  # shape: (145*145, 48)

    if ica:
    # Applica ICA per ridurre ulteriormente a 16 componenti indipendenti
        n_components_ica = n_bands
        ica = FastICA(n_components=n_components_ica, random_state=0)
        data_reduced = ica.fit_transform(data_reduced)  # shape: (145*145, 16)
    
    # Rimodella i dati ridotti nella forma spaziale originale
    data_reduced = data_reduced.reshape(data.shape[0], data.shape[1], n_components_pca)
    data_reduced = data_reduced.transpose(2, 0, 1)  # shape: (16, 145, 145)

    # Stampa le forme
    print(f"Forma originale dei dati: {data.shape}")
    print(f"Forma dopo PCA + ICA: {data_reduced.shape}")

    return data_reduced


class HSI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_windows, augmented = False):
        """
        Args:
            data_windows (list): Lista di tuple (finestra, label) generata da create_dataset.
        """
        self.data_windows = data_windows
        self.augmented = augmented

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):

        window, window_2, window_3, label = self.data_windows[idx]
        # Converti la finestra in float32 per l'input del modello
        windows = [window, window_2, window_3]

        windows = [
            w.float() if w is not None else None
            for w in windows
        ]

        if self.augmented:
            windows = self.apply_augmentation(windows)

        # Converti il label in LongTensor (per le funzioni di loss come CrossEntropyLoss)
        label = torch.tensor(label, dtype=torch.long)
        return windows, label

    def apply_augmentation(self, windows):
        # x1, x2: (C, H, W)
        if random.random() < 0.5:
            windows = [
                torch.flip(w, dims=[1]) if w is not None else None
                for w in windows
            ]
        if random.random() < 0.5:
            windows = [
                torch.flip(w, dims=[2]) if w is not None else None
                for w in windows
        ]
        rot_k = random.choice([0, 1, 2, 3])  # 0°, 90°, 180°, 270°
        windows = [
            torch.rot90(w, k=rot_k, dims=[1,2]) if w is not None else None
            for w in windows
        ]
        return windows


def split_by_percentage(labels, prob):
    non_zero_labels_indices = np.where(labels.numpy() != 0)

    # Combina gli indici di riga e colonna in una lista di coordinate (riga, colonna)
    non_zero_labels_coords = list(zip(non_zero_labels_indices[0], non_zero_labels_indices[1]))

    print(f"Numero totale di labels diversi da 0: {len(non_zero_labels_coords)}")

    # 2) Prendere il 10% dei labels diversi da 10 in modo randomico
    # Calcola il 10% del numero totale di labels diversi da 0
    num_samples = int(len(non_zero_labels_coords) * prob)

    # Seleziona randomicamente gli indici
    random_non_zero_labels_coords = random.sample(non_zero_labels_coords, num_samples)

    unique_classes = torch.unique(labels[labels != 0])
    selected_coords_set = set(random_non_zero_labels_coords)

    for cls in unique_classes.tolist():
        # Trova tutti i pixel appartenenti a quella classe
        cls_indices = np.where(labels.numpy() == cls)
        cls_coords = list(zip(cls_indices[0], cls_indices[1]))

        # Filtra quelli già selezionati
        already_selected = [(r, c) for (r, c) in cls_coords if (r, c) in selected_coords_set]

        if len(already_selected) < 3:
            # Quanti ne mancano per arrivare a 3
            needed = 3 - len(already_selected)
            # Pixel ancora disponibili della classe
            remaining = list(set(cls_coords) - selected_coords_set)
            # Seleziona in modo randomico quelli mancanti
            extra = random.sample(remaining, min(needed, len(remaining)))
            selected_coords_set.update(extra)


    mask = torch.zeros_like(labels, dtype=torch.bool)

    # Imposta a True i pixel selezionati casualmente
    for row, col in selected_coords_set:
      mask[row, col] = True
    
    return mask


def split_by_fixed_numbers(labels, number):
    # Supponiamo che labels sia un tensore PyTorch
    labels_np = labels.numpy()

    # Ottieni le classi uniche diverse da 0
    unique_classes = np.unique(labels_np)
    unique_classes = unique_classes[unique_classes != 0]

    # Crea una maschera vuota
    mask = torch.zeros_like(labels, dtype=torch.bool)

    for cls in unique_classes:
        # Trova gli indici (riga, colonna) dei pixel di questa classe
        cls_indices = np.where(labels_np == cls)
        cls_coords = list(zip(cls_indices[0], cls_indices[1]))

        # Se ci sono meno di 100 pixel in quella classe, prendili tutti
        n_samples = min(number, len(cls_coords))

        # Seleziona randomicamente fino a 100 pixel
        selected_coords = random.sample(cls_coords, n_samples)

        # Imposta a True nella maschera
        for row, col in selected_coords:
            mask[row, col] = True

    print("Forma della maschera:", mask.shape)
    print("Numero totale di True nella maschera:", mask.sum().item())

    return mask


def create_windows(image, image_2, image_3, mask, labels, window_size=16):
    """
    Crea un dataset di finestre di immagini centrate sui pixel della maschera.

    Args:
        image (torch.Tensor): L'immagine iperspettrale ridotta (H, W, C).
        mask (torch.Tensor): La maschera booleana (H, W).
        window_size (int): La dimensione della finestra (window_size x window_size).

    Returns:
        list: Una lista di tuple (finestra, label).
    """
    dataset = []
    c, h, w = image.shape
    pad = window_size // 2  # Padding necessario per centrare il pixel

    # Applica un padding all'immagine per gestire i bordi
    # Il padding è necessario su altezza e larghezza, e su tutti i canali (si usa 0 per i canali)
    if image != None:
        padded_image = torch.nn.functional.pad(image, (pad, pad, pad, pad), "replicate")
    if image_2 != None:
        padded_image_2 = torch.nn.functional.pad(image_2, (pad, pad, pad, pad), "replicate")
    if image_3 != None:
        padded_image_3 = torch.nn.functional.pad(image_3, (pad, pad, pad, pad), "replicate")
    print("Padded image shape: ", padded_image.shape)
    print("Padded image shape2: ", padded_image_2.shape)

    # Trova le coordinate (riga, colonna) dei pixel True nella maschera
    rows, cols = torch.where(mask)

    for r, c in zip(rows, cols):
        windows = [None, None, None]
        # Calcola le coordinate della finestra nell'immagine con padding
        # Il pixel di interesse (r, c) nell'immagine originale corrisponde a (r + pad, c + pad) nell'immagine con padding
        start_row = r + pad - pad
        end_row = r + pad + pad
        start_col = c + pad - pad
        end_col = c + pad + pad

        # Estrai la finestra dall'immagine con padding
        if image is not None:
            windows[0] = padded_image[:, start_row:end_row, start_col:end_col]
        if image_2 != None:
            windows[1] = padded_image_2[:, start_row:end_row, start_col:end_col]
        if image_3 != None:
            windows[2] = padded_image_3[:, start_row:end_row, start_col:end_col]
        
        # Ottieni il label dal tensore originale delle etichette
        label = labels[r, c].item() - 1

        # Aggiungi la finestra e il label al dataset
        dataset.append((windows[0], windows[1], windows[2], label))

    return dataset


def create_datasets(data, hor_info, vert_info, vol_info, split_mode, labels, window_size):
    new_data = [None, None, None]

    if hor_info[0] != 0:
        hor_data = reduce_data_bands(data, hor_info[0], hor_info[1])
        new_data[0] = torch.from_numpy(hor_data)
    
    if vert_info[0] != 0:
        vert_data = reduce_data_bands(data, vert_info[0], vert_info[1])
        new_data[1] = torch.from_numpy(vert_data)
    
    if vol_info[0] != 0:
        vol_data = reduce_data_bands(data, vol_info[0] * 4, vol_info[1])
        new_data[2] = torch.from_numpy(vol_data)
    
    if split_mode[0] == True:
        mask = split_by_percentage(labels, split_mode[1])
    else:
        mask = split_by_fixed_numbers(labels, split_mode[1])
    
    windows = create_windows(new_data[0], new_data[1], new_data[2], mask, labels, window_size)
    hsi_train_dataset = HSI_Dataset(windows, augmented=True)

    test_mask = ~mask
    test_mask_filtered = test_mask & (labels > 0)
    test_dataset_windows = create_windows(new_data[0], new_data[1], new_data[2], test_mask_filtered, window_size)

    n_val = int(len(test_dataset_windows) * 0.1)
    indices = np.random.permutation(len(test_dataset_windows))

    val_idx = indices[:n_val]
    test_idx = indices[n_val:]

    val_dataset = [test_dataset_windows[i] for i in val_idx]
    test_dataset = [test_dataset_windows[i] for i in test_idx]

    print(f"Numero di esempi nel test set: {len(test_dataset)}")

    # 4) Creare un'istanza della classe HSI_Dataset per il test set
    hsi_test_dataset = HSI_Dataset(test_dataset)
    hsi_val_dataset = HSI_Dataset(val_dataset)

    return hsi_train_dataset, hsi_test_dataset, hsi_val_dataset

