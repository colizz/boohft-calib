def year_nano_label(cfg):
    year = str(cfg.year)
    nano_version = getattr(cfg, "nano_version", None)
    if nano_version in (None, ""):
        return year
    nano_version = str(nano_version)
    if not nano_version.startswith("v"):
        nano_version = "v" + nano_version
    return f"{year}_{nano_version}"


def routine_output_name(cfg):
    return f"{cfg.routine_name}_{year_nano_label(cfg)}"
