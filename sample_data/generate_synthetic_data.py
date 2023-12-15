#!/usr/bin/env python
"""Synthetic Data Generation
This notebook generates some simple synthetic data for us to use to demonstrate the ESGPT pipeline. We'll generate a few files:

  1. ``subjects.csv``, which contains static data about each subject.
  2. ``admission_vitals.csv``, which contains records of admissions, transfers, and vitals signs.
  3. ``lab_tests.csv``, which contains records of lab test measurements.

This is all synthetic data designed solely for demonstrating this pipeline. It is not real data, derived from real data, or designed to mimic real data in any way other than plausible file structure.
"""

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import random
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from EventStream.utils import hydra_dataclass

@hydra_dataclass
class GenerateConfig:
    """Parameters for generating synthetic data.

    Args:
        n_subjects: The number of subjects worth of data to generate.
        seed: The random seed to use.
        out_dir: Where to store the synthetic data.
    """

    n_subjects: int = 100
    seed: int = 1
    out_dir: str = "./sample_data/raw"

def make_subjects_df(cfg: GenerateConfig) -> pl.DataFrame:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    BASE_BIRTH_DATE = datetime(1980, 1, 1)
    EYE_COLORS = ["BROWN", "BLUE", "HAZEL", "GREEN", "OTHER"]
    EYE_COLOR_P = [0.45, 0.27, 0.18, 0.09, 0.01]

    def yrs_to_dob(yrs: np.ndarray) -> list[str]:
        return [(BASE_BIRTH_DATE + timedelta(days=365 * x)).strftime("%m/%d/%Y") for x in yrs]

    size = (cfg.n_subjects,)
    subject_data = pl.DataFrame(
        {
            "MRN": np.random.randint(low=14221, high=1578208, size=size),
            "dob": yrs_to_dob(np.random.uniform(low=-10, high=10, size=size)),
            "eye_color": list(np.random.choice(EYE_COLORS, size=size, replace=True, p=EYE_COLOR_P)),
            "height": list(np.random.uniform(low=152.4, high=182.88, size=size)),
        }
    ).sample(fraction=1, with_replacement=False, shuffle=True, seed=1)

    assert len(subject_data["MRN"].unique()) == cfg.n_subjects

    return subject_data

def make_admissions_vitals_df(cfg: GenerateConfig, subject_data: pl.DataFrame) -> tuple[pl.DataFrame, dict[int, list[tuple[datetime, datetime]]]]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    admit_vitals_data = {
        "MRN": [],
        "admit_date": [],
        "disch_date": [],
        "department": [],
        "vitals_date": [],
        "HR": [],
        "temp": [],
    }

    BASE_ADMIT_DATE = datetime(2010, 1, 1)

    hrs = 60
    days = 24 * hrs
    months = 30 * days

    size = (cfg.n_subjects,)
    n_admissions_L = np.random.randint(low=1, high=4, size=size)
    admit_depts_L = np.random.choice(["PULMONARY", "CARDIAC", "ORTHOPEDIC"], size=size, replace=True)

    admissions_by_subject = {}

    for MRN, n_admissions, dept in zip(subject_data["MRN"], n_admissions_L, admit_depts_L):
        admit_gaps = np.random.uniform(low=1 * days, high=6 * months, size=(n_admissions,))
        admit_lens = np.random.uniform(low=12 * hrs, high=14 * days, size=(n_admissions,))

        running_end = BASE_ADMIT_DATE
        admissions_by_subject[MRN] = []

        for gap, L in zip(admit_gaps, admit_lens):
            running_start = running_end + timedelta(minutes=gap)
            running_end = running_start + timedelta(minutes=L)

            admissions_by_subject[MRN].append((running_start, running_end))

            vitals_time = running_start

            running_HR = np.random.uniform(low=60, high=180)
            running_temp = np.random.uniform(low=95, high=101)
            while vitals_time < running_end:
                admit_vitals_data["MRN"].append(MRN)
                admit_vitals_data["admit_date"].append(running_start.strftime("%m/%d/%Y, %H:%M:%S"))
                admit_vitals_data["disch_date"].append(running_end.strftime("%m/%d/%Y, %H:%M:%S"))
                admit_vitals_data["department"].append(dept)
                admit_vitals_data["vitals_date"].append(vitals_time.strftime("%m/%d/%Y, %H:%M:%S"))

                running_HR += np.random.uniform(low=-10, high=10)
                if running_HR < 30: running_HR = 30
                if running_HR > 300: running_HR = 300

                running_temp += np.random.uniform(low=-0.4, high=0.4)
                if running_temp < 95: running_temp = 95
                if running_temp > 104: running_temp = 104

                admit_vitals_data["HR"].append(round(running_HR, 1))
                admit_vitals_data["temp"].append(round(running_temp, 1))

                if 7 < vitals_time.hour < 21:
                    vitals_gap = 30 + np.random.uniform(low=-30, high=30)
                else:
                    vitals_gap = 3 * hrs + np.random.uniform(low=-30, high=30)

                vitals_time += timedelta(minutes=vitals_gap)

    return pl.DataFrame(admit_vitals_data).sample(
        fraction=1, with_replacement=False, shuffle=True, seed=1
    ), admissions_by_subject

def make_labs_df(cfg: GenerateConfig, admissions_by_subject: dict[int, list[tuple[datetime, datetime]]]) -> pl.DataFrame:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    labs_data = {
        "MRN": [],
        "timestamp": [],
        "lab_name": [],
        "lab_value": [],
    }

    def lab_delta_fn(running_vals: dict[str, float], lab_to_meas: str) -> float:
        do_outlier = np.random.uniform() < 0.0001

        if lab_to_meas not in ("GCS", "SOFA") and do_outlier:
            return 1e6

        old_val = running_vals[lab_to_meas]
        if lab_to_meas == "SOFA":
            delta = np.random.randint(low=-2, high=2)
            new_val = old_val + delta
            if new_val < 1:
                new_val = 1
            elif new_val > 4:
                new_val = 4
        elif lab_to_meas == "GCS":
            delta = np.random.randint(low=-4, high=4)
            new_val = old_val + delta
            if new_val < 1:
                new_val = 1
            elif new_val > 15:
                new_val = 15
        elif lab_to_meas == "SpO2":
            delta = np.random.randint(low=-2, high=2)
            new_val = old_val + delta
            if new_val < 50:
                new_val = 50
            elif new_val > 100:
                new_val = 100
        else:
            delta = np.random.uniform(low=-0.1, high=0.1)
            new_val = old_val + delta
            if new_val < 0:
                new_val = 0

        running_vals[lab_to_meas] = new_val
        return round(new_val, 2)


    hrs = 60
    days = 24 * hrs
    months = 30 * days

    for MRN, admissions in admissions_by_subject.items():
        lab_ps = np.random.dirichlet(alpha=[0.1 for _ in range(5)])

        base_lab_gaps = {
            "potassium": np.random.uniform(low=1 * hrs, high=48 * hrs),
            "creatinine": np.random.uniform(low=1 * hrs, high=48 * hrs),
            "SOFA": np.random.uniform(low=1 * hrs, high=48 * hrs),
            "GCS": np.random.uniform(low=1 * hrs, high=48 * hrs),
            "SpO2": np.random.uniform(low=15, high=1 * hrs),
        }

        for st, end in admissions:
            running_lab_values = {
                "potassium": np.random.uniform(low=3, high=6),
                "creatinine": np.random.uniform(low=0.4, high=1.5),
                "SOFA": np.random.randint(low=1, high=4),
                "GCS": np.random.randint(low=1, high=15),
                "SpO2": np.random.randint(low=70, high=100),
            }

            for lab in base_lab_gaps.keys():
                gap = base_lab_gaps[lab]
                labs_time = st + timedelta(minutes=gap + np.random.uniform(low=-30, high=30))

                while labs_time < end:
                    labs_data["MRN"].append(MRN)
                    labs_data["timestamp"].append(labs_time.strftime("%H:%M:%S-%Y-%m-%d"))
                    labs_data["lab_name"].append(lab)

                    labs_data["lab_value"].append(lab_delta_fn(running_lab_values, lab))

                    if 7 < labs_time.hour < 21:
                        labs_gap = gap + np.random.uniform(low=-30, high=30)
                    else:
                        labs_gap = min(2 * gap, 12 * hrs) + np.random.uniform(low=-30, high=30)

                    labs_time += timedelta(minutes=labs_gap)

    return pl.DataFrame(labs_data).sample(fraction=1, with_replacement=False, shuffle=True, seed=1)

def make_medications_data(
    cfg, admissions_by_subject: dict[int, list[tuple[datetime, datetime]]]
) -> pl.DataFrame:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    medications_data = {
        "MRN": [],
        "timestamp": [],
        "name": [],
        "dose": [],
        "frequency": [],
        "duration": [],
        "generic_name": [],
    }

    hrs = 60
    days = 24 * hrs
    months = 30 * days

    med_options = pl.DataFrame({
        'name':    ['Motrin', 'Advil', 'Tylenol', 'Benadryl', 'motrin'],
        'generic': ['Ibuprofen', 'Ibuprofen', 'Acetaminophen', 'Diphenydramine', 'Ibuprofen'],
        'dose_range': [(400, 800), (400, 800), (325, 625), (25, 100), (400, 800)],
        'frequency': [(1, 3), (1, 3), (1, 5), (1, 2), (1, 3)],
        'duration': [(1, 10), (1, 10), (1, 3), (1, 21), (3, 10)],
    })

    for MRN, admissions in admissions_by_subject.items():
        medication_ps = np.random.dirichlet(alpha=[0.1 for _ in range(len(med_options))])

        for st, end in admissions:
            n_meds_taken = np.random.choice(5, 1, p=[0.4, 0.4, 0.1, 0.075, 0.025])
            meds_taken = np.random.choice(med_options['name'].to_list(), n_meds_taken, p=medication_ps)

            for medication in meds_taken:
                med_record = med_options.filter(pl.col('name') == medication).to_dict()

                gap = np.random.uniform(low=2*days, high=14*days)
                medications_time = st + timedelta(minutes=gap + np.random.uniform(low=-30, high=30))

                while medications_time < end:
                    medications_data["MRN"].append(MRN)
                    medications_data["timestamp"].append(medications_time.strftime("%H:%M:%S-%Y-%m-%d"))
                    medications_data["name"].append(medication)
                    medications_data["generic_name"].append(med_record['generic'][0])

                    dose = round((np.random.uniform(*med_record['dose_range'][0])/100))*100
                    duration = np.random.randint(*med_record['duration'][0])
                    frequency = np.random.randint(*med_record['frequency'][0])

                    medications_data["dose"].append(dose)
                    medications_data["frequency"].append(f"{frequency}x/day")
                    medications_data["duration"].append(f"{duration} days")

                    end_time = medications_time + timedelta(days=duration)
                    new_gap = np.random.uniform(low=2*days, high=14*days)

                    medications_time = end_time + timedelta(minutes=new_gap)

    return pl.DataFrame(medications_data).sample(fraction=1, with_replacement=False, shuffle=True, seed=1)

@hydra.main(version_base=None, config_name="generate_config")
def main(cfg: GenerateConfig):
    n_subjects = cfg.n_subjects
    out_dir = Path(cfg.out_dir)
    seed = cfg.seed

    st = datetime.now()
    subjects_data = make_subjects_df(cfg)
    subjects_data.write_csv(out_dir / "subjects.csv")
    print(f"subjects.csv {subjects_data.shape} written to {out_dir} in {datetime.now() - st}:")
    print(subjects_data.head(3))

    st = datetime.now()
    admit_vitals_data, admissions_by_subject = make_admissions_vitals_df(cfg, subjects_data)
    admit_vitals_data.write_csv(out_dir / "admit_vitals.csv")
    print(f"admit_vitals.csv {admit_vitals_data.shape} written to {out_dir} in {datetime.now() - st}:")
    print(admit_vitals_data.head(3))

    st = datetime.now()
    labs_data = make_labs_df(cfg, admissions_by_subject)
    labs_data.write_csv(out_dir / "labs.csv")
    print(f"labs.csv {labs_data.shape} written to {out_dir} in {datetime.now() - st}:")
    print(labs_data.head(3))

    st = datetime.now()
    medications_data = make_medications_data(cfg, admissions_by_subject)
    medications_data.write_csv(out_dir / "medications.csv")
    print(f"medications.csv {medications_data.shape} written to {out_dir} in {datetime.now() - st}:")
    print(medications_data.head(3))

if __name__ == "__main__":
    main()
