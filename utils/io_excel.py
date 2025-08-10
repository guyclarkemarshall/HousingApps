import io, re
import numpy as np
import pandas as pd

HEADLINE_PATTERN = re.compile(r"Proportion of respondents who report.*\((TP\d+)\)", re.IGNORECASE)
SHEETS = [
    ("TSM24_LCRA_Perception", "LCRA"),
    ("TSM24_LCHO_Perception", "LCHO"),
]

def load_tsm_from_excel(file_bytes: bytes) -> pd.DataFrame:
    """Return tidy DF with headline proportions only."""
    def load_sheet(sheet: str, tenure: str) -> pd.DataFrame:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=2)

        id_cols = [
            "Landlord name","Landlord code","Landlord type","Landlord predominant region",
            "Landlord group size (homes owned, LCRA and LCHO combined)",
            "Relevant\xa0tenant population size (LCRA)","Relevant\xa0tenant population size (LCHO)",
            "Required minimum sample size2,3","Total sample size achieved"
        ]
        present = [c for c in id_cols if c in df.columns]
        core = df[present].copy().rename(columns={
            "Landlord name":"landlord_name",
            "Landlord code":"landlord_code",
            "Landlord type":"landlord_type",
            "Landlord predominant region":"region",
            "Landlord group size (homes owned, LCRA and LCHO combined)":"group_size",
            "Relevant\xa0tenant population size (LCRA)":"relevant_population_lcra",
            "Relevant\xa0tenant population size (LCHO)":"relevant_population_lcho",
            "Required minimum sample size2,3":"required_min_sample",
            "Total sample size achieved":"sample_achieved",
        })
        # apply tenure-specific relevant population
        core["relevant_population"] = np.where(
            tenure=="LCRA", core.get("relevant_population_lcra"),
            core.get("relevant_population_lcho")
        )

        metric_cols = {}
        for c in df.columns:
            if isinstance(c, str) and "for each survey method" not in c:
                m = HEADLINE_PATTERN.search(c)
                if m:
                    metric_cols[c] = m.group(1)
        if not metric_cols:
            return pd.DataFrame()

        long = df[list(metric_cols.keys())].melt(var_name="metric_label", value_name="proportion")
        id_rep = pd.concat([core]*len(metric_cols), ignore_index=True)
        out = pd.concat([id_rep, long], axis=1)
        out["metric_code"] = out["metric_label"].map(metric_cols)
        out["tenure"] = tenure

        for c in ["group_size","relevant_population","required_min_sample","sample_achieved","proportion"]:
            if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")

        out = out.dropna(subset=["proportion"])
        out = out[~out["landlord_name"].isna()].reset_index(drop=True)
        return out[[
            "tenure","landlord_name","landlord_code","landlord_type","region","group_size",
            "relevant_population","required_min_sample","sample_achieved","metric_code","metric_label","proportion"
        ]]

    frames = []
    for sheet, tenure in SHEETS:
        frames.append(load_sheet(sheet, tenure))
    tidy = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return tidy
