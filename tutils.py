import torch
from transformers import AutoTokenizer

encoder_model = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(encoder_model)
from transformers import get_constant_schedule_with_warmup

mconer_grouped = {
    'CW': {
        "id": 0,
        "fine": {
            k: v.strip() for k, v in
            enumerate("VisualWork, MusicalWork, WrittenWork, ArtWork, Software, OtherCW".split(','))
        }
    },
    'LOC': {
        "id": 1,
        "fine": {
            k: v.strip() for k, v in
            enumerate("Facility, OtherLOC, HumanSettlement, Station".split(','))
        }
    },
    'GRP': {
        "id": 2,
        "fine": {
            k: v.strip() for k, v in
            enumerate("MusicalGRP, PublicCorp, PrivateCorp, OtherCorp, AerospaceManufacturer, SportsGRP, "
                      "CarManufacturer, TechCorp, ORG".split(','))
        }
    },
    'PER': {
        "id": 3,
        "fine": {
            k: v.strip() for k, v in
            enumerate("Scientist, Artist, Athlete, Politician, Cleric, SportsManager, OtherPER".split(','))
        }
    },
    'PROD': {
        "id": 4,
        "fine": {
            k: v.strip() for k, v in
            enumerate("Clothing, Vehicle, Food, Drink, OtherPROD".split(','))
        }
    },
    'MED': {
        "id": 5,
        "fine": {
            k: v.strip() for k, v in
            enumerate("Medication/Vaccine, MedicalProcedure, AnatomicalStructure, Symptom, Disease".split(','))
        }
    },
    'O': {
        'id': 6,
        'fine': {
            k: v.strip() for k, v in enumerate(['O'])
        }
    }
}


# mconern = {'B-AerospaceManufacturer': 0, 'I-AerospaceManufacturer': 1, 'B-AnatomicalStructure': 2,
#            'I-AnatomicalStructure': 3,
#            'B-ArtWork': 4, 'I-ArtWork': 5, 'B-Artist': 6, 'I-Artist': 7, 'B-Athlete': 8, 'I-Athlete': 9,
#            'B-CarManufacturer': 10, 'I-CarManufacturer': 11, 'B-Cleric': 12, 'I-Cleric': 13, 'B-Clothing': 14,
#            'I-Clothing': 15, 'B-Disease': 16, 'I-Disease': 17, 'B-Drink': 18, 'I-Drink': 19, 'B-Facility': 20,
#            'I-Facility': 21, 'B-Food': 22, 'I-Food': 23, 'B-HumanSettlement': 24, 'I-HumanSettlement': 25,
#            'B-MedicalProcedure': 26,
#            'I-MedicalProcedure': 27, 'B-Medication/Vaccine': 28, 'I-Medication/Vaccine': 29, 'B-MusicalGRP': 30,
#            'I-MusicalGRP': 31,
#            'B-MusicalWork': 32, 'I-MusicalWork': 33, 'O': 34, 'B-ORG': 35, 'I-ORG': 36, 'B-OtherLOC': 37,
#            'I-OtherLOC': 38, 'B-OtherPER': 39,
#            'I-OtherPER': 40, 'B-OtherPROD': 41, 'I-OtherPROD': 42, 'B-Politician': 43, 'I-Politician': 44,
#            'B-PrivateCorp': 45, 'I-PrivateCorp': 46,
#            'B-PublicCorp': 47, 'I-PublicCorp': 48, 'B-Scientist': 49, 'I-Scientist': 50, 'B-Software': 51,
#            'I-Software': 52, 'B-SportsGRP': 53,
#            'I-SportsGRP': 54, 'B-SportsManager': 55, 'I-SportsManager': 56, 'B-Station': 57, 'I-Station': 58,
#            'B-Symptom': 59, 'I-Symptom': 60, 'B-Vehicle': 61,
#            'I-Vehicle': 62, 'B-VisualWork': 63, 'I-VisualWork': 64, 'B-WrittenWork': 65, 'I-WrittenWork': 66}


def indvidual(grouped, fine=True):
    if fine:
        x = {}
        idx = 0
        for key, val in grouped.items():
            if key != 'O':
                for _key, _val in val['fine'].items():
                    x[f"B-{_val}"] = idx
                    idx += 1
                    x[f"I-{_val}"] = idx
                    idx += 1
            else:
                x[key] = idx
                idx += 1
    else:
        x = {}
        idx = 0
        for key, val in grouped.items():
            if key != 'O':
                x[f"B-{key}"] = idx
                idx += 1
                x[f"I-{key}"] = idx
                idx += 1
            else:
                x[key] = idx
                idx += 1
    return x


def invert(grouped):
    x = {}
    for key, val in grouped.items():
        for _key, _val in val['fine'].items():
            if key != 'O':
                x[f"B-{_val}"] = f"B-{key}"
                x[f"I-{_val}"] = f"I-{key}"
            else:
                x['O'] = 'O'
    return x


# wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7,
#             'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}


def get_optimizer(net, opt=False, warmup=10):
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=0.03)
    if opt:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
        return [optimizer], [scheduler]
    return [optimizer]
