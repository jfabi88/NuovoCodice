from scipy.io import loadmat


def load_ip():
    info = {}

    # Carica l'immagine iperspettrale
    data = loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']  # shape: (145, 145, 200)

    # Carica le etichette
    gt = loadmat('Indian_pines_gt.mat')['indian_pines_gt']  # shape: (145, 145)

    info["data_class_names"] = [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
        "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
        "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
                        ]

    info["pre_name"] = "ip"
    info["n_classes"] = 16

    return data, gt, info


def load_ksc():
    info = {}

    # Carica l'immagine iperspettrale
    ksc_data = loadmat('KSC_corrected.mat')
    data = ksc_data['KSC']

    # Carica le etichette
    ksc_labels = loadmat('KSC_gt.mat')
    gt = ksc_labels['KSC_gt']

    info["data_class_names"] = ["Scrub", "Willow swamp", "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock", "Hardwood swamp", "Graminoid marsh",
                        "Spartina marsh", "Cattail marsh", "Salt marsh", "Mud flats", "Water"]

    info["pre_name"] = "ksc"
    info["n_classes"] = 13

    return data, gt, info


def load_sa():
    info = {}

    ksc_data = loadmat('Salinas_corrected.mat')
    data = ksc_data['salinas_corrected']

    # Carica le etichette
    ksc_labels = loadmat('Salinas_gt.mat')
    gt = ksc_labels['salinas_gt']

    info["data_class_names"] = [
        "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
        "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
        "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk",
        "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"]

    info["pre_name"] = "salinas"
    info["n_classes"] = 16

    return data, gt, info


def load_pu():
    info = {}

    ksc_data = loadmat('PaviaU.mat')
    data = ksc_data['paviaU']

    # Carica le etichette
    ksc_labels = loadmat('PaviaU_gt.mat')
    gt = ksc_labels['paviaU_gt']

    info["data_class_names"] = ["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets	", "Bare Soil	", "Bitumen", "Self-Blocking Bricks", "Shadows"]

    info["pre_name"] = "pavia"
    info["n_classes"] = 9

    return data, gt, info


def load_dataset(name):
    if name == "ip":
        return load_ip()
    elif name == "ksc":
        return load_ksc()
    elif name == "sa":
        return load_sa()
    elif name == "pu":
        return load_pu()
