"""
Générateur de données d'entraînement étendu pour l'Oracle de la Guilde.

Ce script génère un grand nombre de cas variés pour améliorer l'entraînement du modèle.
Il crée des données équilibrées avec différents profils d'aventuriers.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Seed pour reproductibilité
np.random.seed(42)


def calculate_survival_probability(force, intelligence, agilite, chance, 
                                   experience, niveau_quete, equipement, fatigue):
    """
    Calcule la probabilité de survie basée sur les stats de l'aventurier.
    
    Logique métier :
    - Force, Intelligence, Agilité contribuent positivement
    - Chance apporte un bonus aléatoire
    - Expérience et niveau d'équipement augmentent les chances
    - Niveau de quête augmente la difficulté
    - Fatigue diminue les performances
    """
    # Score de base (moyenne des stats principales)
    base_score = (force + intelligence + agilite) / 3
    
    # Bonus d'expérience (0 à 20 points)
    exp_bonus = experience * 0.8
    
    # Bonus d'équipement (0 à 50 points)
    equip_bonus = equipement * 0.5
    
    # Malus de fatigue (0 à -50 points)
    fatigue_malus = fatigue * 0.5
    
    # Difficulté basée sur le niveau de quête (1-10)
    difficulty_threshold = 30 + niveau_quete * 4  # 34 à 70
    
    # Score final
    final_score = base_score + exp_bonus + equip_bonus - fatigue_malus
    
    # La chance peut faire basculer le résultat
    luck_factor = (chance - 50) * 0.3  # -15 à +15
    final_score += luck_factor
    
    # Probabilité de survie (sigmoïde)
    prob = 1 / (1 + np.exp(-(final_score - difficulty_threshold) / 15))
    
    return prob


def generate_random_adventurer():
    """Génère un aventurier aléatoire standard."""
    return {
        'force': np.random.uniform(10, 100),
        'intelligence': np.random.uniform(10, 100),
        'agilite': np.random.uniform(10, 100),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(0, 100),
        'fatigue': np.random.uniform(0, 100)
    }


def generate_warrior():
    """Génère un guerrier (force élevée)."""
    return {
        'force': np.random.uniform(70, 100),
        'intelligence': np.random.uniform(20, 50),
        'agilite': np.random.uniform(30, 60),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(40, 100),  # Bonne armure
        'fatigue': np.random.uniform(0, 80)
    }


def generate_mage():
    """Génère un mage (intelligence élevée)."""
    return {
        'force': np.random.uniform(15, 40),
        'intelligence': np.random.uniform(75, 100),
        'agilite': np.random.uniform(30, 60),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(20, 70),  # Équipement magique modéré
        'fatigue': np.random.uniform(0, 100)
    }


def generate_rogue():
    """Génère un voleur (agilité élevée)."""
    return {
        'force': np.random.uniform(30, 60),
        'intelligence': np.random.uniform(40, 70),
        'agilite': np.random.uniform(75, 100),
        'chance': np.random.uniform(30, 100),  # Souvent chanceux
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(20, 60),  # Équipement léger
        'fatigue': np.random.uniform(0, 70)
    }


def generate_paladin():
    """Génère un paladin (équilibré force/intelligence)."""
    return {
        'force': np.random.uniform(55, 85),
        'intelligence': np.random.uniform(55, 85),
        'agilite': np.random.uniform(35, 55),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(5, 25),  # Plus expérimenté
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(50, 100),  # Très bien équipé
        'fatigue': np.random.uniform(0, 60)
    }


def generate_ranger():
    """Génère un ranger (équilibré agilité/force)."""
    return {
        'force': np.random.uniform(50, 75),
        'intelligence': np.random.uniform(40, 65),
        'agilite': np.random.uniform(60, 90),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(30, 70),
        'fatigue': np.random.uniform(10, 80)
    }


def generate_novice():
    """Génère un novice (faibles stats, peu d'expérience)."""
    return {
        'force': np.random.uniform(10, 40),
        'intelligence': np.random.uniform(10, 40),
        'agilite': np.random.uniform(10, 40),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 3),
        'niveau_quete': np.random.randint(1, 5),  # Quêtes faciles
        'equipement': np.random.uniform(0, 30),
        'fatigue': np.random.uniform(0, 50)
    }


def generate_veteran():
    """Génère un vétéran (bonnes stats, beaucoup d'expérience)."""
    return {
        'force': np.random.uniform(50, 90),
        'intelligence': np.random.uniform(50, 90),
        'agilite': np.random.uniform(50, 90),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(15, 25),
        'niveau_quete': np.random.randint(5, 11),  # Quêtes difficiles
        'equipement': np.random.uniform(60, 100),
        'fatigue': np.random.uniform(20, 80)
    }


def generate_lucky():
    """Génère un chanceux (chance très élevée)."""
    return {
        'force': np.random.uniform(20, 60),
        'intelligence': np.random.uniform(20, 60),
        'agilite': np.random.uniform(20, 60),
        'chance': np.random.uniform(85, 100),  # Très chanceux
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(10, 70),
        'fatigue': np.random.uniform(0, 100)
    }


def generate_unlucky():
    """Génère un malchanceux (chance très basse)."""
    return {
        'force': np.random.uniform(40, 80),
        'intelligence': np.random.uniform(40, 80),
        'agilite': np.random.uniform(40, 80),
        'chance': np.random.uniform(0, 15),  # Très malchanceux
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(20, 80),
        'fatigue': np.random.uniform(0, 100)
    }


def generate_exhausted():
    """Génère un aventurier épuisé (fatigue élevée)."""
    return {
        'force': np.random.uniform(40, 90),
        'intelligence': np.random.uniform(40, 90),
        'agilite': np.random.uniform(40, 90),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(20, 80),
        'fatigue': np.random.uniform(85, 100)  # Très fatigué
    }


def generate_fresh():
    """Génère un aventurier reposé (fatigue basse)."""
    return {
        'force': np.random.uniform(30, 80),
        'intelligence': np.random.uniform(30, 80),
        'agilite': np.random.uniform(30, 80),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(20, 80),
        'fatigue': np.random.uniform(0, 15)  # Bien reposé
    }


def generate_well_equipped():
    """Génère un aventurier très bien équipé."""
    return {
        'force': np.random.uniform(30, 70),
        'intelligence': np.random.uniform(30, 70),
        'agilite': np.random.uniform(30, 70),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(90, 100),  # Équipement légendaire
        'fatigue': np.random.uniform(0, 100)
    }


def generate_poorly_equipped():
    """Génère un aventurier mal équipé."""
    return {
        'force': np.random.uniform(30, 80),
        'intelligence': np.random.uniform(30, 80),
        'agilite': np.random.uniform(30, 80),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(0, 10),  # Presque rien
        'fatigue': np.random.uniform(0, 100)
    }


def generate_glass_cannon():
    """Génère un glass cannon (très fort mais fragile)."""
    return {
        'force': np.random.uniform(80, 100),
        'intelligence': np.random.uniform(80, 100),
        'agilite': np.random.uniform(15, 35),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(10, 40),
        'fatigue': np.random.uniform(40, 100)
    }


def generate_tank():
    """Génère un tank (endurant mais lent)."""
    return {
        'force': np.random.uniform(70, 95),
        'intelligence': np.random.uniform(20, 50),
        'agilite': np.random.uniform(10, 30),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(5, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(70, 100),  # Armure lourde
        'fatigue': np.random.uniform(0, 50)
    }


def generate_speedster():
    """Génère un speedster (ultra rapide)."""
    return {
        'force': np.random.uniform(25, 50),
        'intelligence': np.random.uniform(40, 70),
        'agilite': np.random.uniform(90, 100),
        'chance': np.random.uniform(30, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(10, 50),  # Équipement léger
        'fatigue': np.random.uniform(0, 60)
    }


def generate_balanced():
    """Génère un aventurier parfaitement équilibré."""
    base = np.random.uniform(45, 65)
    variation = 5
    return {
        'force': base + np.random.uniform(-variation, variation),
        'intelligence': base + np.random.uniform(-variation, variation),
        'agilite': base + np.random.uniform(-variation, variation),
        'chance': np.random.uniform(40, 60),
        'experience': np.random.uniform(8, 15),
        'niveau_quete': np.random.randint(4, 8),
        'equipement': base + np.random.uniform(-variation, variation),
        'fatigue': np.random.uniform(30, 50)
    }


def generate_prodigy():
    """Génère un prodige (jeune avec excellentes stats)."""
    return {
        'force': np.random.uniform(70, 95),
        'intelligence': np.random.uniform(75, 100),
        'agilite': np.random.uniform(70, 95),
        'chance': np.random.uniform(50, 100),
        'experience': np.random.uniform(0, 5),  # Peu d'expérience
        'niveau_quete': np.random.randint(5, 11),  # Mais prend des risques
        'equipement': np.random.uniform(30, 70),
        'fatigue': np.random.uniform(0, 60)
    }


def generate_old_master():
    """Génère un vieux maître (stats moyennes mais très expérimenté)."""
    return {
        'force': np.random.uniform(35, 60),  # Force déclinante
        'intelligence': np.random.uniform(70, 95),  # Sagesse
        'agilite': np.random.uniform(25, 50),  # Moins agile
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(22, 25),  # Maximum d'expérience
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(50, 90),  # Bon équipement
        'fatigue': np.random.uniform(40, 90)  # Se fatigue plus vite
    }


def generate_overconfident():
    """Génère un aventurier trop confiant (prend des quêtes trop dures)."""
    return {
        'force': np.random.uniform(30, 60),
        'intelligence': np.random.uniform(30, 60),
        'agilite': np.random.uniform(30, 60),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 10),
        'niveau_quete': np.random.randint(8, 11),  # Quêtes trop difficiles
        'equipement': np.random.uniform(20, 60),
        'fatigue': np.random.uniform(0, 100)
    }


def generate_cautious():
    """Génère un aventurier prudent (prend des quêtes faciles)."""
    return {
        'force': np.random.uniform(50, 80),
        'intelligence': np.random.uniform(50, 80),
        'agilite': np.random.uniform(50, 80),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(5, 25),
        'niveau_quete': np.random.randint(1, 4),  # Quêtes faciles
        'equipement': np.random.uniform(40, 100),
        'fatigue': np.random.uniform(0, 50)
    }


def generate_edge_case_high():
    """Génère un cas limite avec toutes les stats maximales."""
    return {
        'force': np.random.uniform(95, 100),
        'intelligence': np.random.uniform(95, 100),
        'agilite': np.random.uniform(95, 100),
        'chance': np.random.uniform(95, 100),
        'experience': np.random.uniform(23, 25),
        'niveau_quete': 10,  # Quête max
        'equipement': np.random.uniform(95, 100),
        'fatigue': np.random.uniform(0, 10)
    }


def generate_edge_case_low():
    """Génère un cas limite avec toutes les stats minimales."""
    return {
        'force': np.random.uniform(10, 15),
        'intelligence': np.random.uniform(10, 15),
        'agilite': np.random.uniform(10, 15),
        'chance': np.random.uniform(10, 15),
        'experience': np.random.uniform(0, 2),
        'niveau_quete': 1,  # Quête min
        'equipement': np.random.uniform(0, 10),
        'fatigue': np.random.uniform(90, 100)
    }


def generate_mixed_high_low():
    """Génère un aventurier avec des stats contrastées."""
    # Alternance haut/bas
    is_force_high = np.random.choice([True, False])
    return {
        'force': np.random.uniform(80, 100) if is_force_high else np.random.uniform(10, 30),
        'intelligence': np.random.uniform(80, 100) if not is_force_high else np.random.uniform(10, 30),
        'agilite': np.random.uniform(10, 30) if is_force_high else np.random.uniform(80, 100),
        'chance': np.random.uniform(10, 100),
        'experience': np.random.uniform(0, 25),
        'niveau_quete': np.random.randint(1, 11),
        'equipement': np.random.uniform(10, 90),
        'fatigue': np.random.uniform(0, 100)
    }


# Liste de tous les générateurs avec leurs poids
GENERATORS = [
    (generate_random_adventurer, 3.0),     # 3x plus fréquent
    (generate_warrior, 1.0),
    (generate_mage, 1.0),
    (generate_rogue, 1.0),
    (generate_paladin, 1.0),
    (generate_ranger, 1.0),
    (generate_novice, 1.0),
    (generate_veteran, 1.0),
    (generate_lucky, 0.8),
    (generate_unlucky, 0.8),
    (generate_exhausted, 0.8),
    (generate_fresh, 0.8),
    (generate_well_equipped, 0.8),
    (generate_poorly_equipped, 0.8),
    (generate_glass_cannon, 0.7),
    (generate_tank, 0.7),
    (generate_speedster, 0.7),
    (generate_balanced, 1.0),
    (generate_prodigy, 0.6),
    (generate_old_master, 0.6),
    (generate_overconfident, 0.8),
    (generate_cautious, 0.8),
    (generate_edge_case_high, 0.3),
    (generate_edge_case_low, 0.3),
    (generate_mixed_high_low, 0.7),
]


def generate_dataset(n_samples: int, noise_level: float = 0.1) -> pd.DataFrame:
    """
    Génère un dataset complet avec n_samples aventuriers.
    
    Args:
        n_samples: Nombre d'échantillons à générer
        noise_level: Niveau de bruit dans la décision de survie (0 à 1)
    
    Returns:
        DataFrame avec les features et labels
    """
    # Extraire générateurs et poids
    generators = [g for g, _ in GENERATORS]
    weights = np.array([w for _, w in GENERATORS])
    weights = weights / weights.sum()  # Normaliser
    
    data = []
    
    for _ in range(n_samples):
        # Choisir un générateur selon les poids
        gen_idx = np.random.choice(len(generators), p=weights)
        generator = generators[gen_idx]
        
        # Générer l'aventurier
        adventurer = generator()
        
        # Calculer la probabilité de survie
        prob = calculate_survival_probability(**adventurer)
        
        # Ajouter du bruit
        prob_with_noise = prob + np.random.uniform(-noise_level, noise_level)
        prob_with_noise = np.clip(prob_with_noise, 0, 1)
        
        # Décider de la survie
        survie = 1 if np.random.random() < prob_with_noise else 0
        
        adventurer['survie'] = survie
        data.append(adventurer)
    
    df = pd.DataFrame(data)
    
    # Réordonner les colonnes
    columns_order = ['force', 'intelligence', 'agilite', 'chance', 
                     'experience', 'niveau_quete', 'equipement', 'fatigue', 'survie']
    df = df[columns_order]
    
    return df


def main():
    # Configuration
    n_train = 10000  # 10x plus de données d'entraînement
    n_val = 2000     # Plus de données de validation aussi
    
    print("="*60)
    print("Génération de données d'entraînement étendues")
    print("="*60)
    
    # Générer les données
    print(f"\n📊 Génération de {n_train} exemples d'entraînement...")
    train_df = generate_dataset(n_train, noise_level=0.1)
    
    print(f"📊 Génération de {n_val} exemples de validation...")
    val_df = generate_dataset(n_val, noise_level=0.1)
    
    # Statistiques
    print(f"\n📈 Statistiques du dataset d'entraînement:")
    print(f"   - Survie=1: {train_df['survie'].sum()} ({100*train_df['survie'].mean():.1f}%)")
    print(f"   - Survie=0: {len(train_df) - train_df['survie'].sum()} ({100*(1-train_df['survie'].mean()):.1f}%)")
    
    print(f"\n📈 Statistiques du dataset de validation:")
    print(f"   - Survie=1: {val_df['survie'].sum()} ({100*val_df['survie'].mean():.1f}%)")
    print(f"   - Survie=0: {len(val_df) - val_df['survie'].sum()} ({100*(1-val_df['survie'].mean()):.1f}%)")
    
    # Sauvegarder
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    train_path = output_dir / "train_extended.csv"
    val_path = output_dir / "val_extended.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ Fichiers sauvegardés:")
    print(f"   - {train_path}")
    print(f"   - {val_path}")
    
    # Afficher quelques exemples
    print(f"\n🔍 Exemples de données générées:")
    print(train_df.head(10).to_string())
    
    # Statistiques détaillées par feature
    print(f"\n📊 Statistiques des features (train):")
    print(train_df.describe().to_string())


if __name__ == "__main__":
    main()
