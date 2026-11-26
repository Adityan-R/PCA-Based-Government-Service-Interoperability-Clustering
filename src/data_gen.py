"""Synthetic government service-usage dataset generator and real data loader.

Problem SIH26129: PCA-Based Dimensionality Reduction for Interoperability Clustering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


CATEGORIES: List[str] = [
    "certificates_vital_stats",
    "pensions_elderly_care",
    "revenue_taxes",
    "health_medical",
    "education_scholarships",
    "land_records_property",
    "utilities_energy",
    "transport_driving",
    "social_welfare_subsidy",
    "employment_labour",
    "housing_urban_dev",
    "agriculture_farmers",
    "business_licensing",
    "police_citizen_safety",
    "public_distribution_food",
]

DOCUMENTS: List[str] = [
    "doc_aadhaar",
    "doc_pan_card",
    "doc_income_certificate",
    "doc_ration_card",
    "doc_land_deed",
    "doc_voter_id",
    "doc_caste_certificate",
    "doc_disability_proof",
]

AUTH_METHODS: List[str] = [
    "auth_mobile_otp",
    "auth_email_otp",
    "auth_aadhaar_biometric",
    "auth_digilocker",
    "auth_username_password",
]

API_CAPABILITIES: List[str] = [
    "api_rest_available",
    "api_webhook_supported",
    "api_open_data_export",
    "api_gov_sdk_integrated",
]

USER_SEGMENTS: List[str] = [
    "user_seg_rural_pct",
    "user_seg_urban_pct",
    "user_seg_senior_citizen_pct",
    "user_seg_youth_pct",
    "user_seg_enterprise_pct",
]

METRICS: List[str] = [
    "metric_monthly_requests",
    "metric_avg_session_duration_sec",
    "metric_repeat_user_rate",
    "metric_peak_hour_load_index",
    "metric_mobile_access_pct",
    "metric_avg_api_latency_ms",
    "metric_transaction_failure_rate",
    "metric_grievance_rate",
    "metric_kiosk_assisted_pct",
    "metric_language_support_count",
]

GRANULAR_SERVICE_FLAGS: List[str] = [
    "flag_birth_certificate",
    "flag_caste_certificate",
    "flag_income_certificate",
    "flag_domicile_certificate",
    "flag_kisan_pm_samman",
    "flag_crop_insurance_pmfby",
    "flag_soil_health_card",
    "flag_old_age_pension",
    "flag_divyang_disability_pension",
    "flag_epfo_passbook_pension",
    "flag_driving_license_renewal",
    "flag_vehicle_rc_transfer",
    "flag_traffic_echallan",
    "flag_land_ror_7_12",
    "flag_property_mutation",
    "flag_stamp_duty_calculator",
    "flag_abha_health_id",
    "flag_telemedicine_opd",
    "flag_electricity_bill_payment",
    "flag_water_sewerage_tax",
    "flag_smart_ration_card",
    "flag_trade_license_fire_noc",
]

GROUND_TRUTH_DEFINITIONS = [
    {
        "cluster_id": 0,
        "name": "Citizen Certificates & Vital Identity Hubs",
        "primary_category": "certificates_vital_stats",
        "secondary_categories": ["social_welfare_subsidy", "public_distribution_food"],
        "policy_action": "Single Sign-On (SSO) & DigiLocker auth sharing",
        "platforms": [
            ("Central e-District National Portal", "Central"),
            ("State e-Seva Citizen Gateway A", "State"),
            ("State e-Seva Citizen Gateway B", "State"),
            ("Municipal Jan Seva Portal North", "Municipal"),
            ("Gramin Citizen Services Kiosk Hub", "State"),
            ("State Unified Certificate Portal", "State"),
        ],
    },
    {
        "cluster_id": 1,
        "name": "Farmer Welfare & Agritech Systems",
        "primary_category": "agriculture_farmers",
        "secondary_categories": ["social_welfare_subsidy", "revenue_taxes"],
        "policy_action": "Complete architectural merger & unified portal",
        "platforms": [
            ("PM-KISAN Direct Benefit Portal", "Central"),
            ("State Krishi Sahayata App", "State"),
            ("Kisan Suvidha Agri Services", "Central"),
            ("Kisan Crop Insurance Hub", "Central"),
            ("State Soil Health & Mandi Gateway", "State"),
            ("Farmer DBT Subsidy Network", "State"),
        ],
    },
    {
        "cluster_id": 2,
        "name": "Pensions & Senior Social Security",
        "primary_category": "pensions_elderly_care",
        "secondary_categories": ["social_welfare_subsidy", "employment_labour"],
        "policy_action": "Single Sign-On (SSO) & shared API gateway",
        "platforms": [
            ("National Pension System Citizen Access", "Central"),
            ("Jeevan Pramaan Digital Life Certificate", "Central"),
            ("State Vridha & Divyang Pension Portal", "State"),
            ("EPFO Member Passbook & Pension", "Central"),
            ("State Social Security Pension Desk", "State"),
            ("Senior Citizen Welfare Registry", "State"),
        ],
    },
    {
        "cluster_id": 3,
        "name": "Transport & Motor Vehicle Regulation",
        "primary_category": "transport_driving",
        "secondary_categories": ["revenue_taxes", "police_citizen_safety"],
        "policy_action": "Complete architectural merger & unified portal",
        "platforms": [
            ("SARATHI Citizen Driving License Portal", "Central"),
            ("VAHAN Vehicle Registration System", "Central"),
            ("State Parivahan Citizen Services", "State"),
            ("City E-Challan & Road Safety Portal", "Municipal"),
            ("State Commercial Transport Gateway", "State"),
            ("Motor Driving Kiosk System", "State"),
        ],
    },
    {
        "cluster_id": 4,
        "name": "Land Records & Revenue Administration",
        "primary_category": "land_records_property",
        "secondary_categories": ["revenue_taxes", "housing_urban_dev"],
        "policy_action": "Shared API gateway & common schema",
        "platforms": [
            ("State Bhulekh Land Records System A", "State"),
            ("State Bhulekh Land Records System B", "State"),
            ("National Land Records Modernization", "Central"),
            ("Property Registration & Stamp Duty Portal", "State"),
            ("Bhoomi Mutation & Encumbrance Hub", "State"),
            ("Urban Land Survey & Cadastral GIS", "Municipal"),
        ],
    },
    {
        "cluster_id": 5,
        "name": "Healthcare & Universal Health Coverage",
        "primary_category": "health_medical",
        "secondary_categories": ["social_welfare_subsidy", "certificates_vital_stats"],
        "policy_action": "Shared API gateway & common schema",
        "platforms": [
            ("Ayushman Bharat ABHA Health Hub", "Central"),
            ("CoWIN Vaccination & Health Passport", "Central"),
            ("e-Sanjeevani National Telemedicine", "Central"),
            ("State Swasthya Bima Health Desk", "State"),
            ("City Hospital OPD Booking System", "Municipal"),
            ("Maternal & Child Health Tracking Portal", "State"),
        ],
    },
    {
        "cluster_id": 6,
        "name": "Municipal Civic Utilities & Energy Services",
        "primary_category": "utilities_energy",
        "secondary_categories": ["revenue_taxes", "housing_urban_dev"],
        "policy_action": "Complete architectural merger & unified portal",
        "platforms": [
            ("Smart City Municipal Bill Desk", "Municipal"),
            ("State Electricity Urja Portal", "State"),
            ("Urban Jal Nigam Water & Sewerage", "Municipal"),
            ("Clean City Civic Grievance Network", "Municipal"),
            ("State Solar Rooftop Subsidy App", "State"),
            ("City Gas & Energy Consumer Portal", "Municipal"),
        ],
    },
]

DISTRACTOR_PLATFORMS = [
    ("State Archaeological & Heritage Tourism Portal", "State", "education_scholarships"),
    ("National Disaster Alert & Siren Network", "Central", "police_citizen_safety"),
    ("Tribal Handicraft & Artisan Direct Market", "Central", "employment_labour"),
    ("Khadi Village Industries e-Bazaar", "Central", "business_licensing"),
    ("Geological Survey Rock & Mineral GeoData", "Central", "business_licensing"),
    ("State Wildlife Sanctuary & Eco-Pass System", "State", "education_scholarships"),
    ("Public Library & Open Knowledge Archive", "Municipal", "education_scholarships"),
]


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load configuration yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_services(
    num_services: int, seed: int
) -> pd.DataFrame:
    """Generate realistic government digital services across categories."""
    rng = np.random.default_rng(seed)
    services_per_cat = max(5, num_services // len(CATEGORIES))
    records = []
    service_id = 1

    descriptions_bank = {
        "certificates_vital_stats": [
            "Online application, verification, and issuance of caste and domicile certificate with Aadhaar biometric validation.",
            "Registration of birth and issuance of digital birth certificates with QR verification.",
            "Legal marriage registration portal with witness e-KYC and digital certificate generation.",
            "Annual family income certificate verification integrated with state revenue databases.",
            "Non-creamy layer OBC certificate processing and digital signature verification.",
        ],
        "pensions_elderly_care": [
            "National social assistance scheme monthly pension disbursal and bank Aadhaar-linkage status.",
            "Digital life certificate verification using facial recognition and iris biometric scanners.",
            "Indira Gandhi national old age pension registration and grievance redressal system.",
            "Divyang disability pension monthly stipend application and medical board verification.",
            "Widow and destitute women social security pension processing with bank DBT.",
        ],
        "revenue_taxes": [
            "Annual urban property tax assessment, payment gateway, and computerized receipt generation.",
            "Commercial trade license fee calculation, professional tax filing, and digital receipt.",
            "Water and sewerage consumption bill payment and tax recovery portal.",
            "Stamp duty valuation calculator and online deed registration appointment scheduling.",
            "State excise and commercial revenue e-challan clearing system.",
        ],
        "health_medical": [
            "Ayushman Bharat PM-JAY electronic health card generation and cashless hospital claim settlement.",
            "Digital outpatient OPD consultation token generation and doctor appointment scheduler.",
            "Telemedicine consultation platform connecting rural health centers to super-specialty doctors.",
            "National immunization tracker and child vaccination schedule notification service.",
            "Emergency blood bank real-time inventory discovery and donor registration network.",
        ],
        "education_scholarships": [
            "Post-matric scholarship portal for SC, ST, and minority students with direct DBT transfer.",
            "Central pre-matric merit scholarship application verification and institute verification.",
            "Primary school online admission portal under Right to Education with randomized lottery.",
            "Academic bank of credits and digital locker degree verification gateway.",
            "Free textbook and educational kit supply tracking system for government schools.",
        ],
        "land_records_property": [
            "Digital computerized Record of Rights (RoR 7/12) viewing, downloading, and QR verification.",
            "Agricultural land online mutation and title transfer tracking with revenue officer approval.",
            "Property non-encumbrance certificate issuance and historical title deed search.",
            "Cadastral land survey maps and satellite parcel boundary visualization service.",
            "Tenancy record verification and agricultural lease agreement registry.",
        ],
        "utilities_energy": [
            "Consumer electricity bill calculation, smart meter top-up, and online billing payment.",
            "New domestic electrical connection application with load feasibility study.",
            "Subsidized LPG cylinder booking confirmation and direct subsidy balance query.",
            "Grid-connected solar rooftop subsidy disbursement and net-metering tracker.",
            "Municipal piped drinking water supply connection and leak reporting portal.",
        ],
        "transport_driving": [
            "Permanent driving license application, slot booking for driving test, and smart card dispatch.",
            "Motor vehicle registration certificate (RC) renewal and address change module.",
            "Real-time e-Challan fine query and digital payment integration with traffic police.",
            "Commercial transport vehicle fitness certificate renewal and permit verification.",
            "Learner driving license online computer test and instantaneous certificate issuance.",
        ],
        "social_welfare_subsidy": [
            "Below Poverty Line (BPL) welfare scheme beneficiary inclusion and e-Ration linkage.",
            "Direct Benefit Transfer (DBT) payment status audit and Aadhaar payment bridge monitoring.",
            "Financial aid for daughters marriage and female education incentive schemes.",
            "Special assistance program for distressed unorganized sector handloom weavers.",
            "Tribal community traditional livelihood grant application and verification.",
        ],
        "employment_labour": [
            "Unorganized worker registration portal (e-Shram) for social security benefits.",
            "Provident fund employee claim settlement, pension passbook, and UAN management.",
            "State employment exchange registration and government job vacancy alert service.",
            "Building and other construction workers (BOCW) welfare board registration.",
            "Apprenticeship training placement matching and monthly stipend subsidy processing.",
        ],
        "housing_urban_dev": [
            "Pradhan Mantri Awas Yojana (PMAY) urban housing beneficiary geo-tagged subsidy tracker.",
            "Automated building architectural plan scrutiny and structural safety approval.",
            "Urban local body municipal building permit and fire safety NOC issuance.",
            "Slum rehabilitation allotment lottery and tenant biometric identification.",
            "Rainwater harvesting incentive approval and urban drainage compliance.",
        ],
        "agriculture_farmers": [
            "PM-KISAN Samman Nidhi installment status and e-KYC bank account seeding tracker.",
            "Pradhan Mantri Fasal Bima Yojana (PMFBY) crop insurance enrollment and damage claim.",
            "Soil Health Card testing recommendation and customized nutrient advisory service.",
            "Real-time agricultural market mandi commodity arrival and wholesale price information.",
            "Subsidized micro-irrigation drip and sprinkler equipment distribution portal.",
        ],
        "business_licensing": [
            "Single Window Clearance System for industrial setting, factory approval, and power sanction.",
            "Shops and commercial establishment online registration and renewal certification.",
            "Food safety FSSAI license application, hygiene inspection, and certification.",
            "Fire and rescue safety certificate (NOC) application and compliance audit.",
            "Pollution control board consent to establish and operate industrial plants.",
        ],
        "police_citizen_safety": [
            "Lost article report generation and digitally authenticated police certificate.",
            "Domestic worker, tenant, and employee police background verification service.",
            "Senior citizen police beat registration and vulnerable citizen emergency SOS app.",
            "Non-cognizable offence e-complaint registration and status tracking.",
            "Cybercrime online complaint filing and financial freeze coordination desk.",
        ],
        "public_distribution_food": [
            "Digital smart ration card application, member addition, and deletion.",
            "National Food Security Act (NFSA) monthly subsidized grain quota allocation query.",
            "Fair Price Shop (FPS) dealer location, stock availability, and electronic PoS audit.",
            "One Nation One Ration Card portability transaction history and biometric authentication.",
            "Grievance redressal for grain quality and fair price shop overcharging.",
        ],
    }

    for cat in CATEGORIES:
        bank = descriptions_bank.get(cat, ["Government citizen digital public service."])
        for i in range(services_per_cat):
            text = bank[i % len(bank)]
            records.append(
                {
                    "service_id": f"SRV_{service_id:04d}",
                    "service_name": f"{cat.replace('_', ' ').title()} Service #{i+1}",
                    "category": cat,
                    "description": text,
                    "complexity_score": float(np.round(rng.uniform(1.0, 5.0), 2)),
                }
            )
            service_id += 1

    df_services = pd.DataFrame(records)
    return df_services


def generate_synthetic_dataset(
    config: dict[str, Any] | None = None,
    save_raw: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate platforms, features, and hidden ground truth labels.

    Returns:
        (platforms_df, services_df, ground_truth_df)
    """
    if config is None:
        config = load_config()

    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    gen_cfg = config.get("data_generation", {})
    num_services = int(gen_cfg.get("num_services", 210))
    missing_rate = float(gen_cfg.get("missing_rate", 0.05))

    df_services = generate_services(num_services=num_services, seed=seed)

    platform_records: List[Dict[str, Any]] = []
    ground_truth_records: List[Dict[str, Any]] = []

    platform_idx = 0

    # 1. Generate ground truth overlap groups
    for group in GROUND_TRUTH_DEFINITIONS:
        cid = group["cluster_id"]
        cname = group["name"]
        primary = group["primary_category"]
        secondaries = group["secondary_categories"]
        policy = group["policy_action"]

        for name, tier in group["platforms"]:
            platform_id = f"PLT_{platform_idx:03d}"
            state = "Central" if tier == "Central" else rng.choice(["Maharashtra", "Karnataka", "Uttar Pradesh", "Tamil Nadu", "Gujarat", "Rajasthan"])

            # Category coverage probabilities
            cat_coverage = {}
            for cat in CATEGORIES:
                if cat == primary:
                    val = rng.uniform(0.70, 0.98)
                elif cat in secondaries:
                    val = rng.uniform(0.35, 0.65)
                else:
                    val = rng.uniform(0.00, 0.15)
                cat_coverage[f"coverage_{cat}"] = float(np.round(val, 3))

            # Service descriptions text bundle
            primary_services = df_services[df_services["category"] == primary]["description"].sample(
                n=min(3, len(df_services[df_services["category"] == primary])),
                random_state=seed + platform_idx,
            ).tolist()
            sec_services = df_services[df_services["category"].isin(secondaries)]["description"].sample(
                n=min(2, len(df_services[df_services["category"].isin(secondaries)])),
                random_state=seed + platform_idx,
            ).tolist()
            text_desc = " ".join(primary_services + sec_services)

            # Document requirements: cluster specific bias
            doc_flags = {}
            for doc in DOCUMENTS:
                if primary in ["certificates_vital_stats", "pensions_elderly_care", "social_welfare_subsidy"]:
                    prob = 0.85 if "aadhaar" in doc or "income" in doc or "ration" in doc else 0.30
                elif primary in ["transport_driving"]:
                    prob = 0.90 if "aadhaar" in doc or "voter" in doc else 0.20
                elif primary in ["land_records_property"]:
                    prob = 0.90 if "land_deed" in doc or "pan" in doc or "aadhaar" in doc else 0.25
                elif primary in ["agriculture_farmers"]:
                    prob = 0.90 if "aadhaar" in doc or "land_deed" in doc or "ration" in doc else 0.20
                else:
                    prob = 0.50
                doc_flags[doc] = int(rng.uniform() < prob)

            # Auth methods: bias
            auth_flags = {}
            for auth in AUTH_METHODS:
                prob = 0.75 if "mobile_otp" in auth or "aadhaar" in auth else 0.40
                auth_flags[auth] = int(rng.uniform() < prob)

            # API capabilities
            api_flags = {}
            for api in API_CAPABILITIES:
                prob = 0.70 if tier == "Central" else 0.45
                api_flags[api] = int(rng.uniform() < prob)

            # User segments (Dirichlet distribution with domain bias)
            if primary == "agriculture_farmers":
                alpha = [6.0, 1.5, 2.0, 1.5, 0.5]
            elif primary == "pensions_elderly_care":
                alpha = [3.0, 3.0, 7.0, 0.5, 0.5]
            elif primary == "transport_driving":
                alpha = [2.0, 5.0, 1.0, 6.0, 2.0]
            elif primary == "utilities_energy":
                alpha = [2.5, 6.0, 2.0, 3.0, 2.0]
            else:
                alpha = [3.5, 3.5, 2.0, 2.5, 1.0]

            seg_weights = rng.dirichlet(alpha)
            seg_dict = {col: float(np.round(seg_weights[k], 4)) for k, col in enumerate(USER_SEGMENTS)}

            # Operational metrics (correlated with platform size & domain)
            req_base = 1500000 if tier == "Central" else (600000 if tier == "State" else 150000)
            req = float(np.round(rng.normal(req_base, req_base * 0.25)))
            req = max(10000.0, req)

            # Correlated metrics
            peak_hour = float(np.round(np.clip(rng.normal(0.65, 0.12), 0.1, 0.99), 3))
            duration = float(np.round(np.clip(rng.normal(240, 60), 60, 600), 1))
            repeat_rate = float(np.round(np.clip(rng.normal(0.55, 0.15), 0.1, 0.95), 3))
            mobile_pct = float(np.round(np.clip(rng.normal(0.68, 0.14), 0.2, 0.95), 3))
            latency = float(np.round(np.clip(rng.normal(320, 90), 80, 1200), 1))
            failure_rate = float(np.round(np.clip(rng.normal(0.04, 0.015), 0.005, 0.15), 4))
            grievance_rate = float(np.round(failure_rate * 1.5 + rng.normal(0.01, 0.005), 4))
            kiosk_pct = float(np.round(np.clip(rng.normal(0.28, 0.12), 0.02, 0.70), 3))
            lang_count = int(rng.choice([1, 2, 3, 11, 22], p=[0.1, 0.3, 0.3, 0.2, 0.1]))

            # Metric dictionary
            metric_dict = {
                "metric_monthly_requests": req,
                "metric_avg_session_duration_sec": duration,
                "metric_repeat_user_rate": repeat_rate,
                "metric_peak_hour_load_index": peak_hour,
                "metric_mobile_access_pct": mobile_pct,
                "metric_avg_api_latency_ms": latency,
                "metric_transaction_failure_rate": failure_rate,
                "metric_grievance_rate": max(0.0, grievance_rate),
                "metric_kiosk_assisted_pct": kiosk_pct,
                "metric_language_support_count": lang_count,
            }

            # Granular service flags with cluster correlation
            service_flags = {}
            for flg in GRANULAR_SERVICE_FLAGS:
                prob = 0.15
                if primary == "certificates_vital_stats" and any(k in flg for k in ["birth", "caste", "income", "domicile"]):
                    prob = 0.88
                elif primary == "agriculture_farmers" and any(k in flg for k in ["kisan", "crop", "soil"]):
                    prob = 0.88
                elif primary == "pensions_elderly_care" and any(k in flg for k in ["old_age", "divyang", "epfo"]):
                    prob = 0.88
                elif primary == "transport_driving" and any(k in flg for k in ["driving", "vehicle", "traffic"]):
                    prob = 0.88
                elif primary == "land_records_property" and any(k in flg for k in ["land", "mutation", "stamp"]):
                    prob = 0.88
                elif primary == "health_medical" and any(k in flg for k in ["abha", "telemedicine"]):
                    prob = 0.88
                elif primary == "utilities_energy" and any(k in flg for k in ["electricity", "water"]):
                    prob = 0.88
                service_flags[flg] = int(rng.uniform() < prob)

            record = {
                "platform_id": platform_id,
                "platform_name": name,
                "tier": tier,
                "state_jurisdiction": state,
                "launch_year": int(rng.integers(2012, 2024)),
                "service_description_text": text_desc,
                "is_synthetic": True,
                **cat_coverage,
                **service_flags,
                **doc_flags,
                **auth_flags,
                **api_flags,
                **seg_dict,
                **metric_dict,
            }
            platform_records.append(record)

            ground_truth_records.append(
                {
                    "platform_id": platform_id,
                    "platform_name": name,
                    "ground_truth_cluster": cid,
                    "cluster_name": cname,
                    "policy_action": policy,
                    "is_distractor": False,
                }
            )
            platform_idx += 1

    # 2. Add distractor / noise platforms
    for name, tier, cat in DISTRACTOR_PLATFORMS:
        platform_id = f"PLT_{platform_idx:03d}"
        state = "Central" if tier == "Central" else rng.choice(["Himachal Pradesh", "Odisha", "Assam", "Goa"])

        cat_coverage = {f"coverage_{c}": float(np.round(rng.uniform(0.02, 0.20), 3)) for c in CATEGORIES}
        cat_coverage[f"coverage_{cat}"] = float(np.round(rng.uniform(0.60, 0.85), 3))

        desc_samples = df_services[df_services["category"] == cat]["description"].sample(
            n=min(2, len(df_services[df_services["category"] == cat])),
            random_state=seed + platform_idx,
        ).tolist()
        text_desc = "Specialized niche administrative domain. " + " ".join(desc_samples)

        service_flags = {flg: int(rng.uniform() < 0.20) for flg in GRANULAR_SERVICE_FLAGS}
        doc_flags = {doc: int(rng.uniform() < 0.35) for doc in DOCUMENTS}
        auth_flags = {auth: int(rng.uniform() < 0.45) for auth in AUTH_METHODS}
        api_flags = {api: int(rng.uniform() < 0.30) for api in API_CAPABILITIES}

        seg_weights = rng.dirichlet([2.0, 2.0, 2.0, 2.0, 2.0])
        seg_dict = {col: float(np.round(seg_weights[k], 4)) for k, col in enumerate(USER_SEGMENTS)}

        req = float(np.round(rng.normal(80000, 25000)))
        metric_dict = {
            "metric_monthly_requests": max(5000.0, req),
            "metric_avg_session_duration_sec": float(np.round(rng.uniform(90, 400), 1)),
            "metric_repeat_user_rate": float(np.round(rng.uniform(0.1, 0.4), 3)),
            "metric_peak_hour_load_index": float(np.round(rng.uniform(0.3, 0.7), 3)),
            "metric_mobile_access_pct": float(np.round(rng.uniform(0.3, 0.7), 3)),
            "metric_avg_api_latency_ms": float(np.round(rng.uniform(200, 800), 1)),
            "metric_transaction_failure_rate": float(np.round(rng.uniform(0.01, 0.08), 4)),
            "metric_grievance_rate": float(np.round(rng.uniform(0.01, 0.05), 4)),
            "metric_kiosk_assisted_pct": float(np.round(rng.uniform(0.05, 0.30), 3)),
            "metric_language_support_count": int(rng.choice([1, 2, 3])),
        }

        record = {
            "platform_id": platform_id,
            "platform_name": name,
            "tier": tier,
            "state_jurisdiction": state,
            "launch_year": int(rng.integers(2015, 2024)),
            "service_description_text": text_desc,
            "is_synthetic": True,
            **cat_coverage,
            **service_flags,
            **doc_flags,
            **auth_flags,
            **api_flags,
            **seg_dict,
            **metric_dict,
        }
        platform_records.append(record)

        ground_truth_records.append(
            {
                "platform_id": platform_id,
                "platform_name": name,
                "ground_truth_cluster": -1,  # Noise / distractor
                "cluster_name": "Niche Administrative Distractor",
                "policy_action": "Maintain independent coexistence with federated discovery",
                "is_distractor": True,
            }
        )
        platform_idx += 1

    df_platforms = pd.DataFrame(platform_records)
    df_ground_truth = pd.DataFrame(ground_truth_records)

    # Inject realistic missing values (5%) across continuous metric and coverage columns
    injectable_cols = [c for c in df_platforms.columns if c.startswith("metric_") or c.startswith("coverage_")]
    for col in injectable_cols:
        mask = rng.uniform(size=len(df_platforms)) < missing_rate
        df_platforms.loc[mask, col] = np.nan

    if save_raw:
        raw_dir = Path(config["paths"]["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        df_platforms.to_csv(config["paths"]["platforms_csv"], index=False)
        df_services.to_csv(config["paths"]["services_csv"], index=False)
        df_ground_truth.to_csv(config["paths"]["ground_truth_csv"], index=False)

    return df_platforms, df_services, df_ground_truth


def load_real_data(csv_path: str | Path) -> pd.DataFrame:
    """Load real platform feature data from an external CSV.

    Validates presence of essential schema columns and sets `is_synthetic = False`.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Real data CSV not found: {path}")

    df = pd.read_csv(path)
    if "platform_id" not in df.columns:
        df["platform_id"] = [f"REAL_PLT_{i:03d}" for i in range(len(df))]
    if "platform_name" not in df.columns:
        df["platform_name"] = [f"Platform {i+1}" for i in range(len(df))]

    # Ensure required category columns exist or default to 0
    for cat in CATEGORIES:
        col = f"coverage_{cat}"
        if col not in df.columns:
            df[col] = 0.0

    if "service_description_text" not in df.columns:
        df["service_description_text"] = ""

    df["is_synthetic"] = False
    return df


if __name__ == "__main__":
    cfg = load_config()
    platforms, services, gt = generate_synthetic_dataset(cfg, save_raw=True)
    print(f"Generated {len(platforms)} platforms, {len(services)} services, {len(gt)} ground truth entries.")
    print(f"Platforms shape: {platforms.shape}")
    print(f"Ground truth clusters: {gt['ground_truth_cluster'].value_counts().to_dict()}")
