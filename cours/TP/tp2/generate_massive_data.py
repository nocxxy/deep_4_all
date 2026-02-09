"""
Générateur MASSIF de données pour l'Oracle de la Guilde (TP2).

Ce script génère un très grand nombre de données avec :
- 50+ profils d'aventuriers différents
- Data augmentation avancée
- Cas adverses et borderline
- Équilibrage des classes

Génère 50 000+ échantillons d'entraînement !
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

np.random.seed(2024)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def clip_stats(adventurer):
    """S'assure que toutes les stats sont dans les bornes valides."""
    adventurer['force'] = np.clip(adventurer['force'], 10, 100)
    adventurer['intelligence'] = np.clip(adventurer['intelligence'], 10, 100)
    adventurer['agilite'] = np.clip(adventurer['agilite'], 10, 100)
    adventurer['chance'] = np.clip(adventurer['chance'], 10, 100)
    adventurer['experience'] = np.clip(adventurer['experience'], 0, 25)
    adventurer['niveau_quete'] = np.clip(adventurer['niveau_quete'], 1, 10)
    adventurer['equipement'] = np.clip(adventurer['equipement'], 0, 100)
    adventurer['fatigue'] = np.clip(adventurer['fatigue'], 0, 100)
    return adventurer


def random_stat(low, high):
    """Génère une stat aléatoire."""
    return np.random.uniform(low, high)


# ============================================================================
# GÉNÉRATEURS DE SURVIVANTS (survie = 1)
# ============================================================================

SURVIVOR_PROFILES = {
    # Profils de classe
    "warrior_tank": lambda: {
        'force': random_stat(75, 100), 'intelligence': random_stat(20, 45),
        'agilite': random_stat(25, 50), 'chance': random_stat(30, 70),
        'experience': random_stat(8, 25), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(70, 100), 'fatigue': random_stat(0, 45), 'survie': 1
    },
    "agile_rogue": lambda: {
        'force': random_stat(30, 55), 'intelligence': random_stat(45, 70),
        'agilite': random_stat(80, 100), 'chance': random_stat(55, 100),
        'experience': random_stat(5, 22), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(30, 65), 'fatigue': random_stat(0, 40), 'survie': 1
    },
    "wise_mage": lambda: {
        'force': random_stat(15, 40), 'intelligence': random_stat(80, 100),
        'agilite': random_stat(35, 60), 'chance': random_stat(40, 80),
        'experience': random_stat(8, 25), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(40, 80), 'fatigue': random_stat(0, 55), 'survie': 1
    },
    "paladin": lambda: {
        'force': random_stat(60, 85), 'intelligence': random_stat(55, 80),
        'agilite': random_stat(40, 60), 'chance': random_stat(40, 70),
        'experience': random_stat(10, 25), 'niveau_quete': np.random.randint(1, 9),
        'equipement': random_stat(55, 90), 'fatigue': random_stat(10, 50), 'survie': 1
    },
    "ranger": lambda: {
        'force': random_stat(50, 75), 'intelligence': random_stat(45, 70),
        'agilite': random_stat(65, 90), 'chance': random_stat(45, 80),
        'experience': random_stat(6, 22), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(35, 70), 'fatigue': random_stat(5, 45), 'survie': 1
    },
    "berserker": lambda: {
        'force': random_stat(85, 100), 'intelligence': random_stat(10, 35),
        'agilite': random_stat(50, 75), 'chance': random_stat(30, 60),
        'experience': random_stat(5, 20), 'niveau_quete': np.random.randint(1, 7),
        'equipement': random_stat(40, 70), 'fatigue': random_stat(0, 35), 'survie': 1
    },
    "monk": lambda: {
        'force': random_stat(55, 75), 'intelligence': random_stat(60, 85),
        'agilite': random_stat(70, 95), 'chance': random_stat(50, 80),
        'experience': random_stat(10, 25), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(20, 50), 'fatigue': random_stat(0, 35), 'survie': 1
    },
    "cleric": lambda: {
        'force': random_stat(35, 55), 'intelligence': random_stat(70, 95),
        'agilite': random_stat(30, 55), 'chance': random_stat(55, 85),
        'experience': random_stat(10, 25), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(50, 80), 'fatigue': random_stat(10, 50), 'survie': 1
    },
    "bard": lambda: {
        'force': random_stat(30, 50), 'intelligence': random_stat(55, 80),
        'agilite': random_stat(55, 80), 'chance': random_stat(70, 100),
        'experience': random_stat(5, 20), 'niveau_quete': np.random.randint(1, 7),
        'equipement': random_stat(30, 60), 'fatigue': random_stat(10, 45), 'survie': 1
    },
    "necromancer": lambda: {
        'force': random_stat(20, 40), 'intelligence': random_stat(85, 100),
        'agilite': random_stat(25, 50), 'chance': random_stat(30, 60),
        'experience': random_stat(12, 25), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(50, 85), 'fatigue': random_stat(20, 55), 'survie': 1
    },
    
    # Profils basés sur les avantages
    "lucky_fool": lambda: {
        'force': random_stat(25, 50), 'intelligence': random_stat(25, 50),
        'agilite': random_stat(25, 50), 'chance': random_stat(88, 100),
        'experience': random_stat(0, 12), 'niveau_quete': np.random.randint(1, 6),
        'equipement': random_stat(20, 55), 'fatigue': random_stat(20, 60), 'survie': 1
    },
    "veteran": lambda: {
        'force': random_stat(45, 70), 'intelligence': random_stat(50, 75),
        'agilite': random_stat(40, 65), 'chance': random_stat(35, 70),
        'experience': random_stat(20, 25), 'niveau_quete': np.random.randint(1, 10),
        'equipement': random_stat(55, 90), 'fatigue': random_stat(25, 65), 'survie': 1
    },
    "well_equipped": lambda: {
        'force': random_stat(35, 60), 'intelligence': random_stat(35, 60),
        'agilite': random_stat(35, 60), 'chance': random_stat(35, 70),
        'experience': random_stat(5, 18), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(88, 100), 'fatigue': random_stat(10, 50), 'survie': 1
    },
    "fresh_start": lambda: {
        'force': random_stat(50, 75), 'intelligence': random_stat(50, 75),
        'agilite': random_stat(50, 75), 'chance': random_stat(45, 80),
        'experience': random_stat(5, 20), 'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(40, 75), 'fatigue': random_stat(0, 18), 'survie': 1
    },
    "easy_quest": lambda: {
        'force': random_stat(35, 65), 'intelligence': random_stat(35, 65),
        'agilite': random_stat(35, 65), 'chance': random_stat(35, 75),
        'experience': random_stat(3, 18), 'niveau_quete': np.random.randint(1, 3),
        'equipement': random_stat(25, 65), 'fatigue': random_stat(15, 65), 'survie': 1
    },
    "all_rounder": lambda: {
        'force': random_stat(65, 90), 'intelligence': random_stat(65, 90),
        'agilite': random_stat(65, 90), 'chance': random_stat(50, 85),
        'experience': random_stat(10, 25), 'niveau_quete': np.random.randint(2, 10),
        'equipement': random_stat(55, 90), 'fatigue': random_stat(0, 45), 'survie': 1
    },
    "prodigy": lambda: {
        'force': random_stat(70, 95), 'intelligence': random_stat(75, 100),
        'agilite': random_stat(70, 95), 'chance': random_stat(55, 90),
        'experience': random_stat(0, 8), 'niveau_quete': np.random.randint(3, 9),
        'equipement': random_stat(35, 70), 'fatigue': random_stat(0, 50), 'survie': 1
    },
    "old_master": lambda: {
        'force': random_stat(35, 60), 'intelligence': random_stat(75, 100),
        'agilite': random_stat(25, 50), 'chance': random_stat(40, 75),
        'experience': random_stat(22, 25), 'niveau_quete': np.random.randint(1, 10),
        'equipement': random_stat(55, 90), 'fatigue': random_stat(40, 80), 'survie': 1
    },
    "cautious": lambda: {
        'force': random_stat(50, 80), 'intelligence': random_stat(55, 85),
        'agilite': random_stat(50, 80), 'chance': random_stat(40, 75),
        'experience': random_stat(8, 25), 'niveau_quete': np.random.randint(1, 4),
        'equipement': random_stat(45, 85), 'fatigue': random_stat(0, 45), 'survie': 1
    },
    
    # Cas limites survivants
    "edge_survivor_lucky": lambda: {
        'force': random_stat(30, 50), 'intelligence': random_stat(30, 50),
        'agilite': random_stat(30, 50), 'chance': random_stat(95, 100),
        'experience': random_stat(15, 25), 'niveau_quete': np.random.randint(1, 5),
        'equipement': random_stat(70, 100), 'fatigue': random_stat(0, 20), 'survie': 1
    },
    "edge_survivor_equipped": lambda: {
        'force': random_stat(40, 60), 'intelligence': random_stat(40, 60),
        'agilite': random_stat(40, 60), 'chance': random_stat(40, 70),
        'experience': random_stat(18, 25), 'niveau_quete': np.random.randint(1, 4),
        'equipement': random_stat(95, 100), 'fatigue': random_stat(0, 25), 'survie': 1
    },
    "survivor_balanced_high": lambda: {
        'force': random_stat(70, 85), 'intelligence': random_stat(70, 85),
        'agilite': random_stat(70, 85), 'chance': random_stat(60, 80),
        'experience': random_stat(12, 22), 'niveau_quete': np.random.randint(4, 9),
        'equipement': random_stat(60, 85), 'fatigue': random_stat(20, 50), 'survie': 1
    },
    "survivor_specialist_str": lambda: {
        'force': random_stat(90, 100), 'intelligence': random_stat(40, 60),
        'agilite': random_stat(40, 60), 'chance': random_stat(50, 80),
        'experience': random_stat(10, 22), 'niveau_quete': np.random.randint(1, 7),
        'equipement': random_stat(60, 90), 'fatigue': random_stat(0, 40), 'survie': 1
    },
    "survivor_specialist_int": lambda: {
        'force': random_stat(40, 60), 'intelligence': random_stat(90, 100),
        'agilite': random_stat(40, 60), 'chance': random_stat(50, 80),
        'experience': random_stat(10, 22), 'niveau_quete': np.random.randint(1, 7),
        'equipement': random_stat(60, 90), 'fatigue': random_stat(0, 40), 'survie': 1
    },
    "survivor_specialist_agi": lambda: {
        'force': random_stat(40, 60), 'intelligence': random_stat(40, 60),
        'agilite': random_stat(90, 100), 'chance': random_stat(50, 80),
        'experience': random_stat(10, 22), 'niveau_quete': np.random.randint(1, 7),
        'equipement': random_stat(50, 80), 'fatigue': random_stat(0, 35), 'survie': 1
    },
}


# ============================================================================
# GÉNÉRATEURS DE NON-SURVIVANTS (survie = 0)
# ============================================================================

DEATH_PROFILES = {
    # Profils de faiblesse
    "weak_novice": lambda: {
        'force': random_stat(10, 35), 'intelligence': random_stat(10, 35),
        'agilite': random_stat(10, 35), 'chance': random_stat(10, 50),
        'experience': random_stat(0, 5), 'niveau_quete': np.random.randint(4, 11),
        'equipement': random_stat(0, 35), 'fatigue': random_stat(40, 85), 'survie': 0
    },
    "overconfident": lambda: {
        'force': random_stat(40, 65), 'intelligence': random_stat(35, 55),
        'agilite': random_stat(40, 60), 'chance': random_stat(20, 50),
        'experience': random_stat(2, 12), 'niveau_quete': np.random.randint(8, 11),
        'equipement': random_stat(20, 55), 'fatigue': random_stat(30, 75), 'survie': 0
    },
    "exhausted": lambda: {
        'force': random_stat(45, 75), 'intelligence': random_stat(45, 75),
        'agilite': random_stat(35, 60), 'chance': random_stat(30, 65),
        'experience': random_stat(5, 20), 'niveau_quete': np.random.randint(4, 11),
        'equipement': random_stat(30, 65), 'fatigue': random_stat(85, 100), 'survie': 0
    },
    "poorly_equipped": lambda: {
        'force': random_stat(45, 70), 'intelligence': random_stat(45, 70),
        'agilite': random_stat(45, 70), 'chance': random_stat(20, 55),
        'experience': random_stat(3, 18), 'niveau_quete': np.random.randint(5, 11),
        'equipement': random_stat(0, 15), 'fatigue': random_stat(40, 85), 'survie': 0
    },
    "unlucky": lambda: {
        'force': random_stat(50, 80), 'intelligence': random_stat(50, 80),
        'agilite': random_stat(50, 80), 'chance': random_stat(0, 18),
        'experience': random_stat(5, 20), 'niveau_quete': np.random.randint(5, 11),
        'equipement': random_stat(30, 65), 'fatigue': random_stat(40, 75), 'survie': 0
    },
    "inexperienced": lambda: {
        'force': random_stat(40, 70), 'intelligence': random_stat(40, 70),
        'agilite': random_stat(40, 70), 'chance': random_stat(25, 60),
        'experience': random_stat(0, 5), 'niveau_quete': np.random.randint(6, 11),
        'equipement': random_stat(25, 60), 'fatigue': random_stat(35, 80), 'survie': 0
    },
    "hard_quest": lambda: {
        'force': random_stat(35, 60), 'intelligence': random_stat(35, 60),
        'agilite': random_stat(35, 60), 'chance': random_stat(20, 55),
        'experience': random_stat(0, 12), 'niveau_quete': 10,
        'equipement': random_stat(20, 55), 'fatigue': random_stat(40, 85), 'survie': 0
    },
    "glass_cannon": lambda: {
        'force': random_stat(80, 100), 'intelligence': random_stat(10, 35),
        'agilite': random_stat(15, 40), 'chance': random_stat(15, 50),
        'experience': random_stat(2, 15), 'niveau_quete': np.random.randint(6, 11),
        'equipement': random_stat(10, 40), 'fatigue': random_stat(55, 90), 'survie': 0
    },
    "one_trick_str": lambda: {
        'force': random_stat(80, 100), 'intelligence': random_stat(15, 35),
        'agilite': random_stat(15, 35), 'chance': random_stat(20, 50),
        'experience': random_stat(3, 15), 'niveau_quete': np.random.randint(6, 11),
        'equipement': random_stat(15, 50), 'fatigue': random_stat(50, 90), 'survie': 0
    },
    "one_trick_int": lambda: {
        'force': random_stat(15, 35), 'intelligence': random_stat(80, 100),
        'agilite': random_stat(15, 35), 'chance': random_stat(20, 50),
        'experience': random_stat(3, 15), 'niveau_quete': np.random.randint(6, 11),
        'equipement': random_stat(15, 50), 'fatigue': random_stat(50, 90), 'survie': 0
    },
    "one_trick_agi": lambda: {
        'force': random_stat(15, 35), 'intelligence': random_stat(15, 35),
        'agilite': random_stat(80, 100), 'chance': random_stat(20, 50),
        'experience': random_stat(3, 15), 'niveau_quete': np.random.randint(6, 11),
        'equipement': random_stat(10, 40), 'fatigue': random_stat(45, 85), 'survie': 0
    },
    "multiple_weaknesses": lambda: {
        'force': random_stat(20, 45), 'intelligence': random_stat(20, 45),
        'agilite': random_stat(20, 45), 'chance': random_stat(10, 40),
        'experience': random_stat(0, 10), 'niveau_quete': np.random.randint(7, 11),
        'equipement': random_stat(5, 35), 'fatigue': random_stat(70, 100), 'survie': 0
    },
    
    # Profils spécifiques qui échouent
    "tired_veteran": lambda: {
        'force': random_stat(50, 70), 'intelligence': random_stat(55, 75),
        'agilite': random_stat(35, 55), 'chance': random_stat(25, 55),
        'experience': random_stat(18, 25), 'niveau_quete': np.random.randint(8, 11),
        'equipement': random_stat(40, 70), 'fatigue': random_stat(80, 100), 'survie': 0
    },
    "arrogant_youth": lambda: {
        'force': random_stat(60, 85), 'intelligence': random_stat(40, 60),
        'agilite': random_stat(60, 80), 'chance': random_stat(15, 40),
        'experience': random_stat(0, 5), 'niveau_quete': np.random.randint(8, 11),
        'equipement': random_stat(25, 55), 'fatigue': random_stat(30, 65), 'survie': 0
    },
    "naked_fighter": lambda: {
        'force': random_stat(70, 95), 'intelligence': random_stat(30, 55),
        'agilite': random_stat(55, 80), 'chance': random_stat(30, 60),
        'experience': random_stat(5, 18), 'niveau_quete': np.random.randint(6, 11),
        'equipement': random_stat(0, 10), 'fatigue': random_stat(35, 70), 'survie': 0
    },
    "cursed": lambda: {
        'force': random_stat(55, 80), 'intelligence': random_stat(55, 80),
        'agilite': random_stat(55, 80), 'chance': random_stat(0, 12),
        'experience': random_stat(8, 20), 'niveau_quete': np.random.randint(6, 11),
        'equipement': random_stat(40, 70), 'fatigue': random_stat(50, 80), 'survie': 0
    },
    "reckless": lambda: {
        'force': random_stat(50, 75), 'intelligence': random_stat(20, 40),
        'agilite': random_stat(50, 75), 'chance': random_stat(25, 55),
        'experience': random_stat(0, 8), 'niveau_quete': 10,
        'equipement': random_stat(20, 50), 'fatigue': random_stat(45, 80), 'survie': 0
    },
    "burnt_out": lambda: {
        'force': random_stat(40, 60), 'intelligence': random_stat(60, 85),
        'agilite': random_stat(30, 50), 'chance': random_stat(20, 50),
        'experience': random_stat(20, 25), 'niveau_quete': np.random.randint(7, 11),
        'equipement': random_stat(35, 60), 'fatigue': random_stat(90, 100), 'survie': 0
    },
    "cheap_gear": lambda: {
        'force': random_stat(50, 75), 'intelligence': random_stat(50, 75),
        'agilite': random_stat(50, 75), 'chance': random_stat(30, 60),
        'experience': random_stat(5, 18), 'niveau_quete': np.random.randint(7, 11),
        'equipement': random_stat(0, 20), 'fatigue': random_stat(50, 85), 'survie': 0
    },
    "death_balanced_low": lambda: {
        'force': random_stat(30, 50), 'intelligence': random_stat(30, 50),
        'agilite': random_stat(30, 50), 'chance': random_stat(25, 50),
        'experience': random_stat(0, 10), 'niveau_quete': np.random.randint(6, 11),
        'equipement': random_stat(20, 50), 'fatigue': random_stat(55, 85), 'survie': 0
    },
    "death_quest_too_hard": lambda: {
        'force': random_stat(45, 65), 'intelligence': random_stat(45, 65),
        'agilite': random_stat(45, 65), 'chance': random_stat(30, 55),
        'experience': random_stat(0, 8), 'niveau_quete': np.random.randint(9, 11),
        'equipement': random_stat(30, 55), 'fatigue': random_stat(50, 80), 'survie': 0
    },
}


# ============================================================================
# CAS ADVERSES (pour robustesse du modèle)
# ============================================================================

ADVERSARIAL_PROFILES = {
    "adversarial_strong_dies": lambda: {
        'force': random_stat(75, 95), 'intelligence': random_stat(70, 90),
        'agilite': random_stat(65, 85), 'chance': random_stat(5, 20),
        'experience': random_stat(0, 6), 'niveau_quete': 10,
        'equipement': random_stat(10, 30), 'fatigue': random_stat(85, 100), 'survie': 0
    },
    "adversarial_weak_survives": lambda: {
        'force': random_stat(20, 40), 'intelligence': random_stat(25, 45),
        'agilite': random_stat(20, 40), 'chance': random_stat(90, 100),
        'experience': random_stat(20, 25), 'niveau_quete': np.random.randint(1, 3),
        'equipement': random_stat(85, 100), 'fatigue': random_stat(0, 15), 'survie': 1
    },
    "adversarial_average_extreme_luck_survives": lambda: {
        'force': random_stat(45, 55), 'intelligence': random_stat(45, 55),
        'agilite': random_stat(45, 55), 'chance': random_stat(95, 100),
        'experience': random_stat(10, 18), 'niveau_quete': np.random.randint(3, 7),
        'equipement': random_stat(50, 70), 'fatigue': random_stat(30, 50), 'survie': 1
    },
    "adversarial_average_extreme_unluck_dies": lambda: {
        'force': random_stat(55, 70), 'intelligence': random_stat(55, 70),
        'agilite': random_stat(55, 70), 'chance': random_stat(0, 10),
        'experience': random_stat(10, 18), 'niveau_quete': np.random.randint(6, 9),
        'equipement': random_stat(40, 60), 'fatigue': random_stat(55, 75), 'survie': 0
    },
}


# ============================================================================
# CAS LIMITES (borderline - 50/50)
# ============================================================================

def generate_borderline():
    """Génère un cas limite avec survie aléatoire."""
    return {
        'force': random_stat(45, 60),
        'intelligence': random_stat(45, 60),
        'agilite': random_stat(45, 60),
        'chance': random_stat(42, 58),
        'experience': random_stat(9, 16),
        'niveau_quete': np.random.randint(4, 7),
        'equipement': random_stat(42, 58),
        'fatigue': random_stat(42, 58),
        'survie': np.random.choice([0, 1])
    }


# ============================================================================
# GÉNÉRATEUR ALÉATOIRE PONDÉRÉ
# ============================================================================

def generate_random_survivor():
    """Génère un survivant aléatoire avec logique basée sur les stats."""
    base = random_stat(55, 90)
    var = random_stat(5, 20)
    return clip_stats({
        'force': base + random_stat(-var, var),
        'intelligence': base + random_stat(-var, var),
        'agilite': base + random_stat(-var, var),
        'chance': random_stat(40, 90),
        'experience': random_stat(8, 25),
        'niveau_quete': np.random.randint(1, 8),
        'equipement': random_stat(45, 95),
        'fatigue': random_stat(0, 50),
        'survie': 1
    })


def generate_random_death():
    """Génère un non-survivant aléatoire avec logique basée sur les stats."""
    base = random_stat(20, 55)
    var = random_stat(5, 20)
    return clip_stats({
        'force': base + random_stat(-var, var),
        'intelligence': base + random_stat(-var, var),
        'agilite': base + random_stat(-var, var),
        'chance': random_stat(10, 55),
        'experience': random_stat(0, 12),
        'niveau_quete': np.random.randint(5, 11),
        'equipement': random_stat(5, 50),
        'fatigue': random_stat(50, 95),
        'survie': 0
    })


# ============================================================================
# AUGMENTATION DE DONNÉES
# ============================================================================

def augment_data(adventurer, noise_level=0.03):
    """Applique une légère perturbation aux données."""
    augmented = adventurer.copy()
    
    for key in ['force', 'intelligence', 'agilite', 'chance', 'equipement', 'fatigue']:
        noise = np.random.uniform(-noise_level * 100, noise_level * 100)
        augmented[key] = np.clip(adventurer[key] + noise, 
                                  0 if key in ['equipement', 'fatigue'] else 10, 100)
    
    noise_exp = np.random.uniform(-noise_level * 25, noise_level * 25)
    augmented['experience'] = np.clip(adventurer['experience'] + noise_exp, 0, 25)
    
    return augmented


# ============================================================================
# GÉNÉRATION DU DATASET COMPLET
# ============================================================================

def generate_massive_dataset(
    n_train: int = 50000,
    n_val: int = 10000,
    augment_factor: float = 0.2
) -> tuple:
    """Génère un dataset massif équilibré."""
    
    survivor_profiles = list(SURVIVOR_PROFILES.values())
    death_profiles = list(DEATH_PROFILES.values())
    adversarial_profiles = list(ADVERSARIAL_PROFILES.values())
    
    def generate_samples(n_samples):
        data = []
        n_per_class = n_samples // 2
        n_borderline = int(n_samples * 0.05)
        n_adversarial = int(n_samples * 0.05)
        n_random = int(n_samples * 0.15)
        n_profiles = n_per_class - n_random // 2 - n_adversarial // 2
        
        # Survivants par profils
        for _ in range(n_profiles):
            profile = np.random.choice(survivor_profiles)
            data.append(clip_stats(profile()))
        
        # Non-survivants par profils
        for _ in range(n_profiles):
            profile = np.random.choice(death_profiles)
            data.append(clip_stats(profile()))
        
        # Cas aléatoires
        for _ in range(n_random // 2):
            data.append(generate_random_survivor())
        for _ in range(n_random // 2):
            data.append(generate_random_death())
        
        # Cas adverses
        for _ in range(n_adversarial):
            profile = np.random.choice(adversarial_profiles)
            data.append(clip_stats(profile()))
        
        # Cas borderline
        for _ in range(n_borderline):
            data.append(generate_borderline())
        
        return data
    
    print(f"  Génération de {n_train} échantillons d'entraînement...")
    train_data = generate_samples(n_train)
    
    # Augmentation des données d'entraînement
    n_augment = int(len(train_data) * augment_factor)
    print(f"  Augmentation: +{n_augment} échantillons...")
    for _ in range(n_augment):
        original = np.random.choice(range(len(train_data)))
        train_data.append(augment_data(train_data[original]))
    
    print(f"  Génération de {n_val} échantillons de validation...")
    val_data = generate_samples(n_val)
    
    # Conversion en DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    # Mélanger
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # Ordre des colonnes
    columns = ['force', 'intelligence', 'agilite', 'chance', 
               'experience', 'niveau_quete', 'equipement', 'fatigue', 'survie']
    train_df = train_df[columns]
    val_df = val_df[columns]
    
    # S'assurer que niveau_quete est float
    train_df['niveau_quete'] = train_df['niveau_quete'].astype(float)
    val_df['niveau_quete'] = val_df['niveau_quete'].astype(float)
    train_df['survie'] = train_df['survie'].astype(int)
    val_df['survie'] = val_df['survie'].astype(int)
    
    return train_df, val_df


def main():
    print("="*70)
    print("🚀 GÉNÉRATEUR MASSIF DE DONNÉES - ORACLE DE LA GUILDE")
    print("="*70)
    
    print(f"\n📋 Profils disponibles:")
    print(f"   - {len(SURVIVOR_PROFILES)} profils de survivants")
    print(f"   - {len(DEATH_PROFILES)} profils de non-survivants")
    print(f"   - {len(ADVERSARIAL_PROFILES)} profils adverses")
    print(f"   - + générateurs aléatoires et borderline")
    
    # Génération
    print("\n📊 Génération en cours...")
    train_df, val_df = generate_massive_dataset(
        n_train=50000,
        n_val=10000,
        augment_factor=0.2
    )
    
    # Statistiques
    print("\n" + "="*50)
    print("📈 STATISTIQUES")
    print("="*50)
    
    print(f"\n🔹 Train: {len(train_df):,} échantillons")
    print(f"   - Survie=1: {train_df['survie'].sum():,} ({100*train_df['survie'].mean():.1f}%)")
    print(f"   - Survie=0: {len(train_df) - train_df['survie'].sum():,} ({100*(1-train_df['survie'].mean()):.1f}%)")
    
    print(f"\n🔹 Val: {len(val_df):,} échantillons")
    print(f"   - Survie=1: {val_df['survie'].sum():,} ({100*val_df['survie'].mean():.1f}%)")
    print(f"   - Survie=0: {len(val_df) - val_df['survie'].sum():,} ({100*(1-val_df['survie'].mean()):.1f}%)")
    
    # Sauvegarde
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    train_path = output_dir / "train_massive.csv"
    val_path = output_dir / "val_massive.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ Fichiers sauvegardés:")
    print(f"   📁 {train_path} ({len(train_df):,} lignes)")
    print(f"   📁 {val_path} ({len(val_df):,} lignes)")
    
    # Remplacer les fichiers par défaut
    print(f"\n🔄 Mise à jour des fichiers par défaut...")
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    print(f"   ✓ train.csv mis à jour")
    print(f"   ✓ val.csv mis à jour")
    
    # Statistiques détaillées
    print("\n" + "="*50)
    print("📊 STATISTIQUES DÉTAILLÉES")
    print("="*50)
    print(train_df.describe().round(2).to_string())
    
    # Corrélations
    print("\n" + "="*50)
    print("🔗 CORRÉLATIONS AVEC LA SURVIE")
    print("="*50)
    correlations = train_df.corr()['survie'].drop('survie').sort_values(ascending=False)
    for feat, corr in correlations.items():
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"   {feat:15s}: {sign}{bar:20s} ({corr:+.3f})")
    
    print("\n" + "="*70)
    print("✨ GÉNÉRATION TERMINÉE !")
    print("="*70)


if __name__ == "__main__":
    main()
