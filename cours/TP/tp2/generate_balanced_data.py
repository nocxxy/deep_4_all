"""
Générateur de données équilibrées pour l'Oracle de la Guilde.

Ce script génère un dataset équilibré (50/50) avec augmentation de données
et des cas adverses pour améliorer la robustesse du modèle.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2024)


def calculate_survival_score(force, intelligence, agilite, chance, 
                             experience, niveau_quete, equipement, fatigue):
    """Calcule un score de survie pour déterminer si l'aventurier survit."""
    # Score combiné des stats principales
    base_score = (force * 0.3 + intelligence * 0.3 + agilite * 0.4) / 100
    
    # Bonus expérience et équipement
    exp_bonus = (experience / 25) * 0.3
    equip_bonus = (equipement / 100) * 0.25
    
    # Malus fatigue et difficulté
    fatigue_malus = (fatigue / 100) * 0.35
    difficulty = (niveau_quete / 10) * 0.4
    
    # Impact de la chance
    luck_factor = ((chance - 50) / 100) * 0.2
    
    final_score = base_score + exp_bonus + equip_bonus - fatigue_malus - difficulty + luck_factor
    return final_score


def generate_survivor(difficulty_modifier=0):
    """Génère un aventurier qui va probablement survivre."""
    base_stat = np.random.uniform(55, 100)
    variation = np.random.uniform(0, 20)
    
    return {
        'force': np.clip(base_stat + np.random.uniform(-variation, variation), 10, 100),
        'intelligence': np.clip(base_stat + np.random.uniform(-variation, variation), 10, 100),
        'agilite': np.clip(base_stat + np.random.uniform(-variation, variation), 10, 100),
        'chance': np.random.uniform(40, 100),
        'experience': np.random.uniform(8, 25),
        'niveau_quete': np.random.randint(1, 7 + difficulty_modifier),
        'equipement': np.random.uniform(40, 100),
        'fatigue': np.random.uniform(0, 50),
        'survie': 1
    }


def generate_non_survivor(difficulty_modifier=0):
    """Génère un aventurier qui va probablement ne pas survivre."""
    base_stat = np.random.uniform(10, 55)
    variation = np.random.uniform(0, 20)
    
    return {
        'force': np.clip(base_stat + np.random.uniform(-variation, variation), 10, 100),
        'intelligence': np.clip(base_stat + np.random.uniform(-variation, variation), 10, 100),
        'agilite': np.clip(base_stat + np.random.uniform(-variation, variation), 10, 100),
        'chance': np.random.uniform(10, 60),
        'experience': np.random.uniform(0, 12),
        'niveau_quete': np.random.randint(5 - difficulty_modifier, 11),
        'equipement': np.random.uniform(0, 50),
        'fatigue': np.random.uniform(50, 100),
        'survie': 0
    }


def generate_borderline_case():
    """Génère un cas limite (50/50 de survie)."""
    return {
        'force': np.random.uniform(40, 60),
        'intelligence': np.random.uniform(40, 60),
        'agilite': np.random.uniform(40, 60),
        'chance': np.random.uniform(40, 60),
        'experience': np.random.uniform(8, 15),
        'niveau_quete': np.random.randint(4, 7),
        'equipement': np.random.uniform(40, 60),
        'fatigue': np.random.uniform(40, 60),
        'survie': np.random.choice([0, 1])
    }


# ============================================================================
# PROFILS SPÉCIAUX - SURVIVANTS
# ============================================================================

def survivor_warrior_tank():
    """Guerrier tank - survit grâce à sa force et son équipement."""
    return {
        'force': np.random.uniform(80, 100),
        'intelligence': np.random.uniform(20, 40),
        'agilite': np.random.uniform(20, 40),
        'chance': np.random.uniform(30, 70),
        'experience': np.random.uniform(10, 25),
        'niveau_quete': np.random.randint(1, 8),
        'equipement': np.random.uniform(75, 100),
        'fatigue': np.random.uniform(0, 40),
        'survie': 1
    }


def survivor_agile_rogue():
    """Voleur agile - survit grâce à son agilité."""
    return {
        'force': np.random.uniform(30, 50),
        'intelligence': np.random.uniform(50, 70),
        'agilite': np.random.uniform(85, 100),
        'chance': np.random.uniform(60, 100),
        'experience': np.random.uniform(5, 20),
        'niveau_quete': np.random.randint(1, 8),
        'equipement': np.random.uniform(30, 60),
        'fatigue': np.random.uniform(0, 35),
        'survie': 1
    }


def survivor_wise_mage():
    """Mage sage - survit grâce à son intelligence."""
    return {
        'force': np.random.uniform(15, 35),
        'intelligence': np.random.uniform(85, 100),
        'agilite': np.random.uniform(40, 60),
        'chance': np.random.uniform(40, 80),
        'experience': np.random.uniform(10, 25),
        'niveau_quete': np.random.randint(1, 8),
        'equipement': np.random.uniform(40, 80),
        'fatigue': np.random.uniform(0, 50),
        'survie': 1
    }


def survivor_lucky_fool():
    """Chanceux fou - survit uniquement grâce à la chance."""
    return {
        'force': np.random.uniform(25, 45),
        'intelligence': np.random.uniform(25, 45),
        'agilite': np.random.uniform(25, 45),
        'chance': np.random.uniform(90, 100),
        'experience': np.random.uniform(0, 10),
        'niveau_quete': np.random.randint(1, 6),
        'equipement': np.random.uniform(20, 50),
        'fatigue': np.random.uniform(20, 60),
        'survie': 1
    }


def survivor_veteran():
    """Vétéran - survit grâce à l'expérience."""
    return {
        'force': np.random.uniform(45, 65),
        'intelligence': np.random.uniform(50, 70),
        'agilite': np.random.uniform(40, 60),
        'chance': np.random.uniform(30, 70),
        'experience': np.random.uniform(20, 25),
        'niveau_quete': np.random.randint(1, 10),
        'equipement': np.random.uniform(60, 90),
        'fatigue': np.random.uniform(30, 60),
        'survie': 1
    }


def survivor_well_equipped():
    """Bien équipé - survit grâce à son équipement légendaire."""
    return {
        'force': np.random.uniform(35, 55),
        'intelligence': np.random.uniform(35, 55),
        'agilite': np.random.uniform(35, 55),
        'chance': np.random.uniform(30, 70),
        'experience': np.random.uniform(5, 15),
        'niveau_quete': np.random.randint(1, 8),
        'equipement': np.random.uniform(90, 100),
        'fatigue': np.random.uniform(0, 40),
        'survie': 1
    }


def survivor_fresh_start():
    """Bien reposé - survit car pas fatigué."""
    return {
        'force': np.random.uniform(50, 70),
        'intelligence': np.random.uniform(50, 70),
        'agilite': np.random.uniform(50, 70),
        'chance': np.random.uniform(40, 80),
        'experience': np.random.uniform(5, 20),
        'niveau_quete': np.random.randint(1, 7),
        'equipement': np.random.uniform(40, 70),
        'fatigue': np.random.uniform(0, 15),
        'survie': 1
    }


def survivor_easy_quest():
    """Quête facile - survit car mission simple."""
    return {
        'force': np.random.uniform(35, 60),
        'intelligence': np.random.uniform(35, 60),
        'agilite': np.random.uniform(35, 60),
        'chance': np.random.uniform(30, 70),
        'experience': np.random.uniform(3, 15),
        'niveau_quete': np.random.randint(1, 3),
        'equipement': np.random.uniform(25, 60),
        'fatigue': np.random.uniform(20, 60),
        'survie': 1
    }


def survivor_paladin():
    """Paladin équilibré - bon partout."""
    return {
        'force': np.random.uniform(60, 80),
        'intelligence': np.random.uniform(60, 80),
        'agilite': np.random.uniform(50, 70),
        'chance': np.random.uniform(40, 70),
        'experience': np.random.uniform(10, 22),
        'niveau_quete': np.random.randint(1, 9),
        'equipement': np.random.uniform(55, 85),
        'fatigue': np.random.uniform(10, 45),
        'survie': 1
    }


def survivor_all_rounder():
    """Polyvalent - excellentes stats partout."""
    return {
        'force': np.random.uniform(70, 90),
        'intelligence': np.random.uniform(70, 90),
        'agilite': np.random.uniform(70, 90),
        'chance': np.random.uniform(50, 80),
        'experience': np.random.uniform(12, 25),
        'niveau_quete': np.random.randint(3, 10),
        'equipement': np.random.uniform(60, 90),
        'fatigue': np.random.uniform(0, 40),
        'survie': 1
    }


# ============================================================================
# PROFILS SPÉCIAUX - NON-SURVIVANTS
# ============================================================================

def death_weak_novice():
    """Novice faible - meurt par manque de compétences."""
    return {
        'force': np.random.uniform(10, 30),
        'intelligence': np.random.uniform(10, 30),
        'agilite': np.random.uniform(10, 30),
        'chance': np.random.uniform(10, 50),
        'experience': np.random.uniform(0, 3),
        'niveau_quete': np.random.randint(3, 11),
        'equipement': np.random.uniform(0, 30),
        'fatigue': np.random.uniform(40, 80),
        'survie': 0
    }


def death_overconfident():
    """Trop confiant - meurt en prenant des risques."""
    return {
        'force': np.random.uniform(40, 60),
        'intelligence': np.random.uniform(30, 50),
        'agilite': np.random.uniform(40, 60),
        'chance': np.random.uniform(20, 50),
        'experience': np.random.uniform(2, 10),
        'niveau_quete': np.random.randint(8, 11),
        'equipement': np.random.uniform(20, 50),
        'fatigue': np.random.uniform(30, 70),
        'survie': 0
    }


def death_exhausted():
    """Épuisé - meurt de fatigue."""
    return {
        'force': np.random.uniform(50, 70),
        'intelligence': np.random.uniform(50, 70),
        'agilite': np.random.uniform(40, 60),
        'chance': np.random.uniform(30, 60),
        'experience': np.random.uniform(5, 20),
        'niveau_quete': np.random.randint(4, 11),
        'equipement': np.random.uniform(30, 60),
        'fatigue': np.random.uniform(85, 100),
        'survie': 0
    }


def death_poorly_equipped():
    """Mal équipé - meurt par manque d'équipement."""
    return {
        'force': np.random.uniform(45, 65),
        'intelligence': np.random.uniform(45, 65),
        'agilite': np.random.uniform(45, 65),
        'chance': np.random.uniform(20, 50),
        'experience': np.random.uniform(3, 15),
        'niveau_quete': np.random.randint(5, 11),
        'equipement': np.random.uniform(0, 15),
        'fatigue': np.random.uniform(40, 80),
        'survie': 0
    }


def death_unlucky():
    """Malchanceux - meurt malgré de bonnes stats."""
    return {
        'force': np.random.uniform(50, 75),
        'intelligence': np.random.uniform(50, 75),
        'agilite': np.random.uniform(50, 75),
        'chance': np.random.uniform(0, 15),
        'experience': np.random.uniform(5, 18),
        'niveau_quete': np.random.randint(5, 11),
        'equipement': np.random.uniform(30, 60),
        'fatigue': np.random.uniform(40, 70),
        'survie': 0
    }


def death_inexperienced():
    """Inexpérimenté - meurt par manque d'expérience."""
    return {
        'force': np.random.uniform(40, 65),
        'intelligence': np.random.uniform(40, 65),
        'agilite': np.random.uniform(40, 65),
        'chance': np.random.uniform(25, 55),
        'experience': np.random.uniform(0, 4),
        'niveau_quete': np.random.randint(6, 11),
        'equipement': np.random.uniform(25, 55),
        'fatigue': np.random.uniform(35, 75),
        'survie': 0
    }


def death_hard_quest():
    """Quête trop difficile - meurt face à la difficulté."""
    return {
        'force': np.random.uniform(35, 55),
        'intelligence': np.random.uniform(35, 55),
        'agilite': np.random.uniform(35, 55),
        'chance': np.random.uniform(20, 50),
        'experience': np.random.uniform(0, 10),
        'niveau_quete': 10,
        'equipement': np.random.uniform(20, 50),
        'fatigue': np.random.uniform(40, 80),
        'survie': 0
    }


def death_glass_cannon():
    """Glass cannon - fort en attaque mais meurt rapidement."""
    return {
        'force': np.random.uniform(75, 95),
        'intelligence': np.random.uniform(10, 30),
        'agilite': np.random.uniform(15, 35),
        'chance': np.random.uniform(15, 45),
        'experience': np.random.uniform(2, 12),
        'niveau_quete': np.random.randint(6, 11),
        'equipement': np.random.uniform(10, 35),
        'fatigue': np.random.uniform(50, 85),
        'survie': 0
    }


def death_one_trick():
    """One trick - bon dans un domaine mais pas assez."""
    stat_type = np.random.choice(['force', 'intelligence', 'agilite'])
    adventurer = {
        'force': np.random.uniform(20, 40),
        'intelligence': np.random.uniform(20, 40),
        'agilite': np.random.uniform(20, 40),
        'chance': np.random.uniform(20, 50),
        'experience': np.random.uniform(3, 12),
        'niveau_quete': np.random.randint(6, 11),
        'equipement': np.random.uniform(15, 45),
        'fatigue': np.random.uniform(50, 85),
        'survie': 0
    }
    adventurer[stat_type] = np.random.uniform(75, 95)
    return adventurer


def death_multiple_weaknesses():
    """Multiples faiblesses - cumul de problèmes."""
    return {
        'force': np.random.uniform(25, 45),
        'intelligence': np.random.uniform(25, 45),
        'agilite': np.random.uniform(25, 45),
        'chance': np.random.uniform(10, 35),
        'experience': np.random.uniform(0, 8),
        'niveau_quete': np.random.randint(7, 11),
        'equipement': np.random.uniform(5, 30),
        'fatigue': np.random.uniform(70, 95),
        'survie': 0
    }


# ============================================================================
# CAS ADVERSES (pour tester la robustesse)
# ============================================================================

def adversarial_strong_but_dies():
    """Adversarial: Fort mais meurt quand même."""
    return {
        'force': np.random.uniform(70, 90),
        'intelligence': np.random.uniform(70, 90),
        'agilite': np.random.uniform(60, 80),
        'chance': np.random.uniform(5, 20),  # Très malchanceux
        'experience': np.random.uniform(0, 5),  # Peu d'expérience
        'niveau_quete': 10,  # Quête max
        'equipement': np.random.uniform(10, 30),  # Mal équipé
        'fatigue': np.random.uniform(80, 100),  # Épuisé
        'survie': 0
    }


def adversarial_weak_but_survives():
    """Adversarial: Faible mais survit quand même."""
    return {
        'force': np.random.uniform(20, 40),
        'intelligence': np.random.uniform(20, 40),
        'agilite': np.random.uniform(20, 40),
        'chance': np.random.uniform(85, 100),  # Très chanceux
        'experience': np.random.uniform(18, 25),  # Très expérimenté
        'niveau_quete': np.random.randint(1, 3),  # Quête facile
        'equipement': np.random.uniform(80, 100),  # Très bien équipé
        'fatigue': np.random.uniform(0, 15),  # Bien reposé
        'survie': 1
    }


# ============================================================================
# GÉNÉRATEUR PRINCIPAL
# ============================================================================

SURVIVOR_GENERATORS = [
    (generate_survivor, 4.0),
    (survivor_warrior_tank, 1.0),
    (survivor_agile_rogue, 1.0),
    (survivor_wise_mage, 1.0),
    (survivor_lucky_fool, 0.8),
    (survivor_veteran, 1.0),
    (survivor_well_equipped, 1.0),
    (survivor_fresh_start, 1.0),
    (survivor_easy_quest, 1.2),
    (survivor_paladin, 1.0),
    (survivor_all_rounder, 0.8),
    (adversarial_weak_but_survives, 0.5),
]

DEATH_GENERATORS = [
    (generate_non_survivor, 4.0),
    (death_weak_novice, 1.2),
    (death_overconfident, 1.0),
    (death_exhausted, 1.0),
    (death_poorly_equipped, 1.0),
    (death_unlucky, 0.8),
    (death_inexperienced, 1.0),
    (death_hard_quest, 1.0),
    (death_glass_cannon, 0.8),
    (death_one_trick, 0.8),
    (death_multiple_weaknesses, 1.0),
    (adversarial_strong_but_dies, 0.5),
]


def generate_balanced_dataset(n_samples: int) -> pd.DataFrame:
    """Génère un dataset équilibré 50/50."""
    
    # Préparer les générateurs
    survivor_gens = [g for g, _ in SURVIVOR_GENERATORS]
    survivor_weights = np.array([w for _, w in SURVIVOR_GENERATORS])
    survivor_weights = survivor_weights / survivor_weights.sum()
    
    death_gens = [g for g, _ in DEATH_GENERATORS]
    death_weights = np.array([w for _, w in DEATH_GENERATORS])
    death_weights = death_weights / death_weights.sum()
    
    data = []
    n_survivors = n_samples // 2
    n_deaths = n_samples - n_survivors
    
    # Générer les survivants
    for _ in range(n_survivors):
        gen_idx = np.random.choice(len(survivor_gens), p=survivor_weights)
        adventurer = survivor_gens[gen_idx]()
        data.append(adventurer)
    
    # Générer les morts
    for _ in range(n_deaths):
        gen_idx = np.random.choice(len(death_gens), p=death_weights)
        adventurer = death_gens[gen_idx]()
        data.append(adventurer)
    
    # Ajouter des cas limites (10%)
    n_borderline = int(n_samples * 0.1)
    for _ in range(n_borderline):
        data.append(generate_borderline_case())
    
    df = pd.DataFrame(data)
    
    # Mélanger le dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # S'assurer que toutes les valeurs sont dans les bornes
    df['force'] = df['force'].clip(10, 100)
    df['intelligence'] = df['intelligence'].clip(10, 100)
    df['agilite'] = df['agilite'].clip(10, 100)
    df['chance'] = df['chance'].clip(10, 100)
    df['experience'] = df['experience'].clip(0, 25)
    df['niveau_quete'] = df['niveau_quete'].clip(1, 10).astype(float)
    df['equipement'] = df['equipement'].clip(0, 100)
    df['fatigue'] = df['fatigue'].clip(0, 100)
    df['survie'] = df['survie'].astype(int)
    
    columns_order = ['force', 'intelligence', 'agilite', 'chance', 
                     'experience', 'niveau_quete', 'equipement', 'fatigue', 'survie']
    return df[columns_order]


def apply_data_augmentation(df: pd.DataFrame, augment_factor: float = 0.2) -> pd.DataFrame:
    """
    Applique une augmentation de données en ajoutant du bruit aux exemples existants.
    """
    n_augmented = int(len(df) * augment_factor)
    
    augmented_data = []
    for _ in range(n_augmented):
        # Choisir un exemple aléatoire
        idx = np.random.randint(len(df))
        example = df.iloc[idx].copy()
        
        # Ajouter du bruit (±5% sur chaque feature sauf survie et niveau_quete)
        noise_features = ['force', 'intelligence', 'agilite', 'chance', 
                          'experience', 'equipement', 'fatigue']
        
        for feat in noise_features:
            noise = np.random.uniform(-5, 5)
            example[feat] = np.clip(example[feat] + noise, 
                                    0 if feat in ['experience', 'equipement', 'fatigue'] else 10,
                                    25 if feat == 'experience' else 100)
        
        augmented_data.append(example)
    
    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([df, augmented_df], ignore_index=True)


def main():
    print("="*70)
    print("Génération de données d'entraînement équilibrées et augmentées")
    print("="*70)
    
    # Configuration
    n_train_base = 8000
    n_val = 2000
    
    # Générer les datasets de base
    print(f"\n📊 Génération de {n_train_base} exemples d'entraînement (base)...")
    train_df = generate_balanced_dataset(n_train_base)
    
    # Appliquer l'augmentation de données (20% de plus)
    print(f"📈 Augmentation des données d'entraînement...")
    train_df = apply_data_augmentation(train_df, augment_factor=0.25)
    
    print(f"📊 Génération de {n_val} exemples de validation...")
    val_df = generate_balanced_dataset(n_val)
    
    # Mélanger à nouveau après augmentation
    train_df = train_df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # Statistiques
    print(f"\n" + "="*50)
    print("📈 STATISTIQUES")
    print("="*50)
    
    print(f"\n🔹 Dataset d'entraînement: {len(train_df)} exemples")
    print(f"   - Survie=1: {train_df['survie'].sum()} ({100*train_df['survie'].mean():.1f}%)")
    print(f"   - Survie=0: {len(train_df) - train_df['survie'].sum()} ({100*(1-train_df['survie'].mean()):.1f}%)")
    
    print(f"\n🔹 Dataset de validation: {len(val_df)} exemples")
    print(f"   - Survie=1: {val_df['survie'].sum()} ({100*val_df['survie'].mean():.1f}%)")
    print(f"   - Survie=0: {len(val_df) - val_df['survie'].sum()} ({100*(1-val_df['survie'].mean()):.1f}%)")
    
    # Sauvegarder
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    train_path = output_dir / "train_balanced.csv"
    val_path = output_dir / "val_balanced.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ Fichiers sauvegardés:")
    print(f"   📁 {train_path}")
    print(f"   📁 {val_path}")
    
    # Stats détaillées
    print(f"\n" + "="*50)
    print("📊 STATISTIQUES DÉTAILLÉES DES FEATURES")
    print("="*50)
    print(train_df.describe().round(2).to_string())
    
    # Corrélations avec la survie
    print(f"\n" + "="*50)
    print("🔗 CORRÉLATIONS AVEC LA SURVIE")
    print("="*50)
    correlations = train_df.corr()['survie'].drop('survie').sort_values(ascending=False)
    for feat, corr in correlations.items():
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"   {feat:15s}: {sign}{bar:20s} ({corr:+.3f})")


if __name__ == "__main__":
    main()
