# -*- coding: utf-8 -*-
"""Google Colab generator for the hard synthetic RECIST evaluation dataset.

This file intentionally retains Colab syntax (`!pip` and `drive.mount`) and is
not intended to be executed as a regular local Python module.
"""

# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
!pip install fpdf2 faker

# ==========================================
# 2. IMPORTS & SETUP
# ==========================================
import os
import json
import random
import shutil
import datetime
from copy import deepcopy
from fpdf import FPDF
from faker import Faker

# Mount Google Drive automatically
from google.colab import drive
drive.mount('/content/drive')

fake = Faker('fr_FR')
RANDOM_SEED = 20260821
random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)

# Local Colab Directories
PDF_DIR = "/content/reports"
FILLED_JSON_DIR = "/content/filtered_templates/filled"
EMPTY_JSON_DIR = "/content/filtered_templates/empty"
EXTRACTED_JSON_DIR = "/content/extracted"

# Google Drive Directories (Fixed path to 'MyDrive')
DRIVE_PDF_DIR = "/content/drive/MyDrive/Synthetic_Body_CT_Reports/reports"
DRIVE_FILLED_DIR = "/content/drive/MyDrive/Synthetic_Body_CT_Reports/filtered_templates/filled"
DRIVE_EMPTY_DIR = "/content/drive/MyDrive/Synthetic_Body_CT_Reports/filtered_templates/empty"
DRIVE_EXTRACTED_DIR = "/content/drive/MyDrive/Synthetic_Body_CT_Reports/extracted"

# Create all directories
for d in [
    PDF_DIR,
    FILLED_JSON_DIR,
    EMPTY_JSON_DIR,
    EXTRACTED_JSON_DIR,
    DRIVE_PDF_DIR,
    DRIVE_FILLED_DIR,
    DRIVE_EMPTY_DIR,
    DRIVE_EXTRACTED_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 3. GLOBAL MASTER SCHEMA (EMPTY TEMPLATE)
# ==========================================
BASE_EMPTY_TEMPLATE = {
    "Imaging_RECIST": {
        "Size_major_nodule_mm_start_AtezoBev_D0": {
            "valeur": None,
            "unité": "mm",
        },
        "Response_at_first_imaging_RECIST": {
            "valeur": None,
        },
    }
}

# Helper to fix unsupported PDF characters
def sanitize_text(text):
    if not text: return ""
    return text.replace("œ", "oe").replace("Œ", "OE").replace("€", "EUR")

# ==========================================
# 4. HARD RECIST SCENARIO GENERATOR
# Gold = baseline TARGET lesion size in mm + official CONCLUSION RECIST code.
# Reports mix paraphrases, extra lesions, current vs baseline sizes, and
# informal wording that can conflict with the conclusion.
# ==========================================
ORGANS = [
    ("foie", "du", ["segment II", "segment IV", "segment VI", "segment VII", "dôme hépatique"]),
    ("poumon", "du", ["lobe supérieur droit", "lobe inférieur gauche", "lingula", "lobe moyen"]),
    ("rein", "du", ["pôle supérieur", "pôle inférieur", "sinus rénal"]),
    ("surrénale", "de la", ["bras interne", "bras externe"]),
    ("péritoine", "du", ["gouttière pariéto-colique droite", "cul-de-sac de Douglas"]),
]
NON_TARGET_KINDS = [
    ("adénopathie médiastinale", "une"),
    ("nodule pulmonaire sous-pleural", "un"),
    ("lésion osseuse lytique", "une"),
    ("implant péritonéal", "un"),
    ("lésion surrénalienne controlatérale", "une"),
]
INFORMAL_RECIST = {
    "CR": ["disparition des cibles", "plus aucune lésion mesurable", "réponse complète clinique"],
    "PR": ["régression franche", "diminution nette des diamètres", "réponse partielle apparente"],
    "SD": ["aspect inchangé", "stabilité morphologique", "pas de modification significative"],
    "PD": ["majoration des lésions", "évolution défavorable", "suspicion de progression"],
    "NE": ["comparaison limitée", "examen difficilement comparable", "qualité d'injection insuffisante"],
}
WRONG_CODES = {
    "CR": ["PD", "SD"],
    "PR": ["PD", "SD"],
    "SD": ["PD", "PR"],
    "PD": ["SD", "PR"],
    "NE": ["SD", "PD"],
}


def _size_text(mm):
    """Paraphrase a measurement; gold remains integer millimetres."""
    cm = mm / 10.0
    return random.choice(
        [
            f"{mm} mm",
            f"{mm} millimètres",
            f"{cm:.1f} cm".replace(".", ","),
            f"{cm:.1f} cm",
            f"{mm} mm de grand axe",
            f"un diamètre maximal de {mm} mm",
            f"environ {mm} mm",
        ]
    )


def _location():
    organ, prep, segments = random.choice(ORGANS)
    return organ, prep, random.choice(segments)


def _official_recist(code):
    """Conclusion phrasing; sometimes omits the latin-letter code."""
    with_code = {
        "CR": [
            "Selon RECIST 1.1, la réponse officielle retenue est une réponse complète (CR).",
            "Catégorie RECIST retenue : CR.",
        ],
        "PR": [
            "Selon RECIST 1.1, la réponse officielle retenue est une réponse partielle (PR).",
            "Catégorie RECIST retenue : PR.",
        ],
        "SD": [
            "Selon RECIST 1.1, la réponse officielle retenue est une maladie stable (SD).",
            "Catégorie RECIST retenue : SD.",
        ],
        "PD": [
            "Selon RECIST 1.1, la réponse officielle retenue est une progression (PD).",
            "Catégorie RECIST retenue : PD.",
        ],
        "NE": [
            "Selon RECIST 1.1, la réponse officielle retenue est non évaluable (NE).",
            "Catégorie RECIST retenue : NE.",
        ],
    }
    without_code = {
        "CR": [
            "Au total, disparition des lésions cibles, compatible avec une réponse complète.",
            "Bilan de fin d'évaluation : réponse complète.",
        ],
        "PR": [
            "Au total, diminution suffisante des lésions cibles pour une réponse partielle.",
            "Bilan de fin d'évaluation : réponse partielle.",
        ],
        "SD": [
            "Au total, variation insuffisante pour conclure à une réponse ou une progression : maladie stable.",
            "Bilan de fin d'évaluation : maladie stable.",
        ],
        "PD": [
            "Au total, critères de progression atteints sur les lésions cibles et/ou apparition de lésion(s) nouvelle(s).",
            "Bilan de fin d'évaluation : progression tumorale.",
        ],
        "NE": [
            "Au total, les critères RECIST ne peuvent pas être appliqués de façon fiable sur cet examen.",
            "Bilan de fin d'évaluation : réponse non évaluable.",
        ],
    }
    pool = with_code[code] + without_code[code]
    return random.choice(pool)


def _distract_size(baseline_mm):
    candidate = baseline_mm + random.choice([-28, -15, -8, 9, 14, 22, 31, 40])
    return max(6, min(120, candidate))


def generate_recist_scenario():
    baseline_size_mm = random.randint(18, 96)
    recist_code = random.choice(["CR", "PR", "SD", "PD", "NE"])
    organ, prep, segment = _location()
    current_size_mm = _distract_size(baseline_size_mm)
    if recist_code == "CR":
        current_size_mm = 0

    n_extra = random.choice([1, 2, 3])
    extras = []
    used = {(organ, segment)}
    for _ in range(n_extra):
        extra_organ, extra_prep, extra_segment = _location()
        if (extra_organ, extra_segment) in used:
            extra_organ, extra_prep, extra_segment = _location()
        used.add((extra_organ, extra_segment))
        kind, kind_art = random.choice(NON_TARGET_KINDS)
        extras.append(
            {
                "kind": kind,
                "kind_art": kind_art,
                "organ": extra_organ,
                "prep": extra_prep,
                "segment": extra_segment,
                "size": _distract_size(baseline_size_mm),
            }
        )

    informal = random.choice(INFORMAL_RECIST[recist_code])
    conflicting = random.choice(INFORMAL_RECIST[random.choice(WRONG_CODES[recist_code])])
    wrong_code = random.choice(WRONG_CODES[recist_code])

    scan_title = random.choice(
        [
            "THORACO-ABDOMINO-PELVIEN ONCOLOGIQUE",
            "TAP AVEC CONTRASTE - SUIVI ONCOLOGIQUE",
            "SCANNER D'ÉVALUATION TUMORALE",
        ]
    )
    indication = random.choice(
        [
            "Suivi sous atezolizumab-bevacizumab. Comparaison au bilan de référence (J0).",
            f"Contrôle oncologique. Le clinicien évoque une {conflicting} ; à confronter aux critères RECIST.",
            f"Réévaluation. Suspicion clinique de {wrong_code} à confirmer ou infirmer.",
            "Bilan de première évaluation d'imagerie après introduction du traitement.",
        ]
    )

    target_baseline = random.choice(
        [
            (
                f"Lésion cible n°1 (référence J0 / D0) : nodule {prep} {organ} "
                f"({segment}), mesuré à {_size_text(baseline_size_mm)} au scanner initial."
            ),
            (
                f"La cible principale retenue au démarrage du traitement siège au {segment} "
                f"({organ}) et mesurait {_size_text(baseline_size_mm)} de plus grand diamètre."
            ),
            (
                f"Au bilan d'inclusion, la lésion cible index ({organ}, {segment}) "
                f"était chiffrée à {_size_text(baseline_size_mm)}."
            ),
        ]
    )

    if recist_code == "CR":
        target_current = random.choice(
            [
                "Sur l'examen actuel, cette cible n'est plus individualisable.",
                "La lésion cible n°1 n'est plus visible, sans reliquat mesurable.",
            ]
        )
    else:
        target_current = random.choice(
            [
                (
                    f"Sur l'examen actuel, la même cible mesure {_size_text(current_size_mm)} "
                    f"(ne pas confondre avec la mesure d'inclusion)."
                ),
                (
                    f"Contrôle du jour : {_size_text(current_size_mm)} pour la lésion cible n°1. "
                    f"La valeur de référence reste celle du scanner D0."
                ),
            ]
        )

    extra_sentences = []
    for extra in extras:
        extra_sentences.append(
            random.choice(
                [
                    (
                        f"Lésion non cible : {extra['kind']} au niveau {extra['prep']} {extra['organ']} "
                        f"({extra['segment']}) mesurant {_size_text(extra['size'])}."
                    ),
                    (
                        f"On note également {extra['kind_art']} {extra['kind']} "
                        f"({extra['organ']}, {extra['segment']}) "
                        f"à {_size_text(extra['size'])}, non retenu(e) comme cible."
                    ),
                ]
            )
        )

    findings_conflict = random.choice(
        [
            f"Les coupes axiales donnent un {conflicting}, sans permettre à elles seules la catégorie finale.",
            f"Impression morphologique initiale : {conflicting}. Cette impression n'est pas la conclusion RECIST.",
            f"Un compte rendu antérieur mentionnait {wrong_code} ; ce n'est pas l'évaluation retenue aujourd'hui.",
        ]
    )

    new_lesion = ""
    if recist_code == "PD" and random.random() < 0.6:
        new_lesion = (
            f"Apparition d'une lésion nouvelle de {_size_text(random.randint(8, 24))} "
            "dans le foie gauche, argument de progression."
        )
    elif recist_code != "PD":
        new_lesion = "Pas de lésion nouvelle indiscutable sur cet examen."

    technique = random.choice(
        [
            "Acquisition hélicoïdale, temps portal, reconstructions millimétriques, comparaison au scanner de référence.",
            "Scanner TAP après injection iodée. Recalage visuel avec l'examen d'inclusion.",
            "Protocole oncologique. Qualité d'examen " + ("limitée" if recist_code == "NE" else "satisfaisante") + ".",
        ]
    )

    recist_body = (
        f"Analyse qualitative : {informal}. {findings_conflict} "
        "La catégorie officielle figure uniquement en conclusion."
    )

    conclusion = _official_recist(recist_code)
    if recist_code == "NE":
        conclusion += " Les mesures restent décrites à titre indicatif."

    sections = {
        "Technique": technique,
        "Lésions cibles": target_baseline + " " + target_current,
        "Lésions non cibles": " ".join(extra_sentences),
        "Analyse intermédiaire": recist_body + ((" " + new_lesion) if new_lesion else ""),
    }
    gt_data = {
        "Imaging_RECIST": {
            "Size_major_nodule_mm_start_AtezoBev_D0": baseline_size_mm,
            "Response_at_first_imaging_RECIST": recist_code,
        }
    }
    return scan_title, indication, sections, conclusion, gt_data

# ==========================================
# 5. FIXED PDF GENERATOR CLASS
# ==========================================
class MedicalReportPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 14)
        self.cell(0, 10, sanitize_text("CENTRE D'IMAGERIE MÉDICALE PARIS"), ln=True, align="C")
        self.set_font("helvetica", "", 10)
        self.cell(0, 5, "Service de Radiologie", ln=True, align="L")
        self.ln(5)

def create_pdf(filename, patient, scan_title, indication, sections, conclusion):
    pdf = MedicalReportPDF()
    pdf.add_page()

    # Title
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "COMPTE RENDU DE SCANNER", ln=True, align="C")
    pdf.cell(0, 5, sanitize_text(scan_title), ln=True, align="C")
    pdf.ln(8)

    # Patient Info Table
    pdf.set_fill_color(240, 240, 240)

    # Row 1
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(35, 8, "Nom", border=1, fill=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(60, 8, sanitize_text(patient['nom']), border=1)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(35, 8, sanitize_text("Prénom"), border=1, fill=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(60, 8, sanitize_text(patient['prenom']), border=1, ln=True)

    # Row 2
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(35, 8, "Date de naissance", border=1, fill=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(60, 8, patient['dob'], border=1)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(35, 8, "Sexe", border=1, fill=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(60, 8, patient['sexe'], border=1, ln=True)

    # Row 3
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(35, 8, sanitize_text("N° Dossier"), border=1, fill=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(60, 8, patient['dossier'], border=1)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(35, 8, "Date examen", border=1, fill=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(60, 8, patient['date_examen'], border=1, ln=True)
    pdf.ln(10)

    # Body: Indication
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, "Indication :", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 6, sanitize_text(indication))
    pdf.ln(4)

    # Body: Results
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, sanitize_text("RÉSULTATS"), ln=True)

    for section_name, text in sections.items():
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, sanitize_text(section_name) + " :", ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_text(text))
        pdf.ln(2)

    # Conclusion
    pdf.ln(4)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "CONCLUSION", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 6, sanitize_text(conclusion))

    # Footer
    pdf.ln(10)
    pdf.cell(0, 6, sanitize_text(f"Radiologue : Dr {fake.last_name()}"), ln=True)
    pdf.cell(0, 6, sanitize_text(f"Date de rédaction : {patient['date_examen']}"), ln=True)

    pdf.output(filename)

# ==========================================
# 6. JSON TEMPLATE GENERATOR
# ==========================================
def generate_jsons(base_name, patient, gt_data):
    # 1. EMPTY TEMPLATE
    empty_template = deepcopy(BASE_EMPTY_TEMPLATE)
    empty_path = os.path.join(EMPTY_JSON_DIR, f"{base_name}_template_empty.json")
    with open(empty_path, "w", encoding="utf-8") as f:
        json.dump(empty_template, f, indent=2, ensure_ascii=False)

    # 2. FILLED TEMPLATE
    filled_template = deepcopy(BASE_EMPTY_TEMPLATE)

    for category, attributes in gt_data.items():
        if category in filled_template:
            for key, val in attributes.items():
                if key in filled_template[category]:
                    filled_template[category][key]["valeur"] = val

    filled_path = os.path.join(FILLED_JSON_DIR, f"{base_name}_template_filled.json")
    with open(filled_path, "w", encoding="utf-8") as f:
        json.dump(filled_template, f, indent=2, ensure_ascii=False)

# ==========================================
# 7. MAIN EXECUTION LOOP
# ==========================================
NUM_SAMPLES = 1000
print(f"Starting generation of {NUM_SAMPLES} PDF & JSON sets...")

for i in range(1, NUM_SAMPLES + 1):
    gender = random.choice(['M', 'F'])
    first_name = fake.first_name_male() if gender == 'M' else fake.first_name_female()
    last_name = fake.last_name().upper()
    dob_obj = fake.date_of_birth(minimum_age=18, maximum_age=90)
    exam_date_obj = fake.date_between(start_date='-2y', end_date='today')

    patient_info = {
        'nom': last_name, 'prenom': first_name, 'dob': dob_obj.strftime("%d/%m/%Y"),
        'sexe': gender, 'dossier': f"{random.randint(100000, 999999)}",
        'date_examen': exam_date_obj.strftime("%d/%m/%Y")
    }

    scan_title, indication, sections, conclusion, gt_data = generate_recist_scenario()

    # Keep gold attributes out of the filename to avoid evaluation leakage.
    base_name = f"sample_{i:04d}"

    create_pdf(os.path.join(PDF_DIR, base_name + ".pdf"), patient_info, scan_title, indication, sections, conclusion)
    generate_jsons(base_name, patient_info, gt_data)

    if i % 100 == 0:
        print(f"Generated {i} out of {NUM_SAMPLES}...")

# ==========================================
# 8. SAFE COPY TO GOOGLE DRIVE (NO METADATA ERRORS)
# ==========================================
print("\nSafely moving files to Google Drive (ignoring metadata to prevent crashes)...")

def safe_copy(src_folder, dst_folder):
    files = os.listdir(src_folder)
    for file_name in files:
        # Using shutil.copy instead of copy2 skips copying strict timestamps which GDrive rejects
        shutil.copy(os.path.join(src_folder, file_name), os.path.join(dst_folder, file_name))

print("Copying PDFs...")
safe_copy(PDF_DIR, DRIVE_PDF_DIR)

print("Copying Filled JSONs...")
safe_copy(FILLED_JSON_DIR, DRIVE_FILLED_DIR)

print("Copying Empty JSONs...")
safe_copy(EMPTY_JSON_DIR, DRIVE_EMPTY_DIR)

print("✅ Complete! All PDFs and JSONs have been successfully created and transferred.")
