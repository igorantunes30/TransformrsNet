"""
Compara as 20 espécies do especie_passaros com as 200 do CUB-200-2011.
Níveis de correspondência: Exata, Mesmo gênero, Mesma família, Nenhuma.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Dados ─────────────────────────────────────────────────────────────────────
# 20 espécies do especie_passaros com taxonomia
ep_species = {
    "ABBOTTS BABBLER":           {"genus": "Malacocincla",   "family": "Pellorneidae"},
    "ABBOTTS BOOBY":             {"genus": "Papasula",       "family": "Sulidae"},
    "ABYSSINIAN GROUND HORNBILL":{"genus": "Bucorvus",       "family": "Bucorvidae"},
    "AFRICAN CROWNED CRANE":     {"genus": "Balearica",      "family": "Gruidae"},
    "AFRICAN EMERALD CUCKOO":    {"genus": "Chrysococcyx",   "family": "Cuculidae"},
    "AFRICAN FIREFINCH":         {"genus": "Lagonosticta",   "family": "Estrildidae"},
    "AFRICAN OYSTER CATCHER":    {"genus": "Haematopus",     "family": "Haematopodidae"},
    "AFRICAN PIED HORNBILL":     {"genus": "Lophoceros",     "family": "Bucerotidae"},
    "AFRICAN PYGMY GOOSE":       {"genus": "Nettapus",       "family": "Anatidae"},
    "ALBATROSS":                 {"genus": "Diomedea",       "family": "Diomedeidae"},
    "ALBERTS TOWHEE":            {"genus": "Melozone",       "family": "Passerellidae"},
    "ALEXANDRINE PARAKEET":      {"genus": "Psittacula",     "family": "Psittaculidae"},
    "ALPINE CHOUGH":             {"genus": "Pyrrhocorax",    "family": "Corvidae"},
    "ALTAMIRA YELLOWTHROAT":     {"genus": "Geothlypis",     "family": "Parulidae"},
    "AMERICAN AVOCET":           {"genus": "Recurvirostra",  "family": "Recurvirostridae"},
    "AMERICAN BITTERN":          {"genus": "Botaurus",       "family": "Ardeidae"},
    "AMERICAN COOT":             {"genus": "Fulica",         "family": "Rallidae"},
    "AMERICAN FLAMINGO":         {"genus": "Phoenicopterus", "family": "Phoenicopteridae"},
    "AMERICAN GOLDFINCH":        {"genus": "Spinus",         "family": "Fringillidae"},
    "AMERICAN KESTREL":          {"genus": "Falco",          "family": "Falconidae"},
}

# 200 espécies CUB com taxonomia (gênero / família)
cub_species = {
    "Black footed Albatross":    {"genus": "Phoebastria",    "family": "Diomedeidae"},
    "Laysan Albatross":          {"genus": "Phoebastria",    "family": "Diomedeidae"},
    "Sooty Albatross":           {"genus": "Phoebetria",     "family": "Diomedeidae"},
    "Groove billed Ani":         {"genus": "Crotophaga",     "family": "Cuculidae"},
    "Crested Auklet":            {"genus": "Aethia",         "family": "Alcidae"},
    "Least Auklet":              {"genus": "Aethia",         "family": "Alcidae"},
    "Parakeet Auklet":           {"genus": "Aethia",         "family": "Alcidae"},
    "Rhinoceros Auklet":         {"genus": "Cerorhinca",     "family": "Alcidae"},
    "Brewer Blackbird":          {"genus": "Euphagus",       "family": "Icteridae"},
    "Red winged Blackbird":      {"genus": "Agelaius",       "family": "Icteridae"},
    "Rusty Blackbird":           {"genus": "Euphagus",       "family": "Icteridae"},
    "Yellow headed Blackbird":   {"genus": "Xanthocephalus", "family": "Icteridae"},
    "Bobolink":                  {"genus": "Dolichonyx",     "family": "Icteridae"},
    "Indigo Bunting":            {"genus": "Passerina",      "family": "Cardinalidae"},
    "Lazuli Bunting":            {"genus": "Passerina",      "family": "Cardinalidae"},
    "Painted Bunting":           {"genus": "Passerina",      "family": "Cardinalidae"},
    "Cardinal":                  {"genus": "Cardinalis",     "family": "Cardinalidae"},
    "Spotted Catbird":           {"genus": "Ailuroedus",     "family": "Ptilonorhynchidae"},
    "Gray Catbird":              {"genus": "Dumetella",      "family": "Mimidae"},
    "Yellow breasted Chat":      {"genus": "Icteria",        "family": "Icteriidae"},
    "Eastern Towhee":            {"genus": "Pipilo",         "family": "Passerellidae"},
    "Chuck will Widow":          {"genus": "Antrostomus",    "family": "Caprimulgidae"},
    "Brandt Cormorant":          {"genus": "Urile",          "family": "Phalacrocoracidae"},
    "Red faced Cormorant":       {"genus": "Urile",          "family": "Phalacrocoracidae"},
    "Pelagic Cormorant":         {"genus": "Urile",          "family": "Phalacrocoracidae"},
    "Bronzed Cowbird":           {"genus": "Molothrus",      "family": "Icteridae"},
    "Shiny Cowbird":             {"genus": "Molothrus",      "family": "Icteridae"},
    "Brown Creeper":             {"genus": "Certhia",        "family": "Certhiidae"},
    "American Crow":             {"genus": "Corvus",         "family": "Corvidae"},
    "Fish Crow":                 {"genus": "Corvus",         "family": "Corvidae"},
    "Black billed Cuckoo":       {"genus": "Coccyzus",       "family": "Cuculidae"},
    "Mangrove Cuckoo":           {"genus": "Coccyzus",       "family": "Cuculidae"},
    "Yellow billed Cuckoo":      {"genus": "Coccyzus",       "family": "Cuculidae"},
    "Gray crowned Rosy Finch":   {"genus": "Leucosticte",    "family": "Fringillidae"},
    "Purple Finch":              {"genus": "Haemorhous",     "family": "Fringillidae"},
    "Northern Flicker":          {"genus": "Colaptes",       "family": "Picidae"},
    "Acadian Flycatcher":        {"genus": "Empidonax",      "family": "Tyrannidae"},
    "Great Crested Flycatcher":  {"genus": "Myiarchus",      "family": "Tyrannidae"},
    "Least Flycatcher":          {"genus": "Empidonax",      "family": "Tyrannidae"},
    "Olive sided Flycatcher":    {"genus": "Contopus",       "family": "Tyrannidae"},
    "Scissor tailed Flycatcher": {"genus": "Tyrannus",       "family": "Tyrannidae"},
    "Vermilion Flycatcher":      {"genus": "Pyrocephalus",   "family": "Tyrannidae"},
    "Yellow bellied Flycatcher": {"genus": "Empidonax",      "family": "Tyrannidae"},
    "Frigatebird":               {"genus": "Fregata",        "family": "Fregatidae"},
    "Northern Fulmar":           {"genus": "Fulmarus",       "family": "Procellariidae"},
    "Gadwall":                   {"genus": "Mareca",         "family": "Anatidae"},
    "American Goldfinch":        {"genus": "Spinus",         "family": "Fringillidae"},
    "European Goldfinch":        {"genus": "Carduelis",      "family": "Fringillidae"},
    "Boat tailed Grackle":       {"genus": "Quiscalus",      "family": "Icteridae"},
    "Eared Grebe":               {"genus": "Podiceps",       "family": "Podicipedidae"},
    "Horned Grebe":              {"genus": "Podiceps",       "family": "Podicipedidae"},
    "Pied billed Grebe":         {"genus": "Podilymbus",     "family": "Podicipedidae"},
    "Western Grebe":             {"genus": "Aechmophorus",   "family": "Podicipedidae"},
    "Blue Grosbeak":             {"genus": "Passerina",      "family": "Cardinalidae"},
    "Evening Grosbeak":          {"genus": "Hesperiphona",   "family": "Fringillidae"},
    "Pine Grosbeak":             {"genus": "Pinicola",       "family": "Fringillidae"},
    "Rose breasted Grosbeak":    {"genus": "Pheucticus",     "family": "Cardinalidae"},
    "Pigeon Guillemot":          {"genus": "Cepphus",        "family": "Alcidae"},
    "California Gull":           {"genus": "Larus",          "family": "Laridae"},
    "Glaucous winged Gull":      {"genus": "Larus",          "family": "Laridae"},
    "Heermann Gull":             {"genus": "Larus",          "family": "Laridae"},
    "Herring Gull":              {"genus": "Larus",          "family": "Laridae"},
    "Ivory Gull":                {"genus": "Pagophila",      "family": "Laridae"},
    "Ring billed Gull":          {"genus": "Larus",          "family": "Laridae"},
    "Slaty backed Gull":         {"genus": "Larus",          "family": "Laridae"},
    "Western Gull":              {"genus": "Larus",          "family": "Laridae"},
    "Anna Hummingbird":          {"genus": "Calypte",        "family": "Trochilidae"},
    "Ruby throated Hummingbird": {"genus": "Archilochus",    "family": "Trochilidae"},
    "Rufous Hummingbird":        {"genus": "Selasphorus",    "family": "Trochilidae"},
    "Green Violetear":           {"genus": "Colibri",        "family": "Trochilidae"},
    "Long tailed Jaeger":        {"genus": "Stercorarius",   "family": "Stercorariidae"},
    "Pomarine Jaeger":           {"genus": "Stercorarius",   "family": "Stercorariidae"},
    "Blue Jay":                  {"genus": "Cyanocitta",     "family": "Corvidae"},
    "Florida Jay":               {"genus": "Aphelocoma",     "family": "Corvidae"},
    "Green Jay":                 {"genus": "Cyanocorax",     "family": "Corvidae"},
    "Dark eyed Junco":           {"genus": "Junco",          "family": "Passerellidae"},
    "Tropical Kingbird":         {"genus": "Tyrannus",       "family": "Tyrannidae"},
    "Gray Kingbird":             {"genus": "Tyrannus",       "family": "Tyrannidae"},
    "Belted Kingfisher":         {"genus": "Megaceryle",     "family": "Alcedinidae"},
    "Green Kingfisher":          {"genus": "Chloroceryle",   "family": "Alcedinidae"},
    "Pied Kingfisher":           {"genus": "Ceryle",         "family": "Alcedinidae"},
    "Ringed Kingfisher":         {"genus": "Megaceryle",     "family": "Alcedinidae"},
    "White breasted Kingfisher": {"genus": "Halcyon",        "family": "Alcedinidae"},
    "Red legged Kittiwake":      {"genus": "Rissa",          "family": "Laridae"},
    "Horned Lark":               {"genus": "Eremophila",     "family": "Alaudidae"},
    "Pacific Loon":              {"genus": "Gavia",          "family": "Gaviidae"},
    "Mallard":                   {"genus": "Anas",           "family": "Anatidae"},
    "Western Meadowlark":        {"genus": "Sturnella",      "family": "Icteridae"},
    "Hooded Merganser":          {"genus": "Lophodytes",     "family": "Anatidae"},
    "Red breasted Merganser":    {"genus": "Mergus",         "family": "Anatidae"},
    "Mockingbird":               {"genus": "Mimus",          "family": "Mimidae"},
    "Nighthawk":                 {"genus": "Chordeiles",     "family": "Caprimulgidae"},
    "Clark Nutcracker":          {"genus": "Nucifraga",      "family": "Corvidae"},
    "White breasted Nuthatch":   {"genus": "Sitta",          "family": "Sittidae"},
    "Baltimore Oriole":          {"genus": "Icterus",        "family": "Icteridae"},
    "Hooded Oriole":             {"genus": "Icterus",        "family": "Icteridae"},
    "Orchard Oriole":            {"genus": "Icterus",        "family": "Icteridae"},
    "Scott Oriole":              {"genus": "Icterus",        "family": "Icteridae"},
    "Ovenbird":                  {"genus": "Seiurus",        "family": "Parulidae"},
    "Brown Pelican":             {"genus": "Pelecanus",      "family": "Pelecanidae"},
    "White Pelican":             {"genus": "Pelecanus",      "family": "Pelecanidae"},
    "Western Wood Pewee":        {"genus": "Contopus",       "family": "Tyrannidae"},
    "Sayornis":                  {"genus": "Sayornis",       "family": "Tyrannidae"},
    "American Pipit":            {"genus": "Anthus",         "family": "Motacillidae"},
    "Whip poor Will":            {"genus": "Antrostomus",    "family": "Caprimulgidae"},
    "Horned Puffin":             {"genus": "Fratercula",     "family": "Alcidae"},
    "Common Raven":              {"genus": "Corvus",         "family": "Corvidae"},
    "White necked Raven":        {"genus": "Corvus",         "family": "Corvidae"},
    "American Redstart":         {"genus": "Setophaga",      "family": "Parulidae"},
    "Geococcyx":                 {"genus": "Geococcyx",      "family": "Cuculidae"},
    "Loggerhead Shrike":         {"genus": "Lanius",         "family": "Laniidae"},
    "Great Grey Shrike":         {"genus": "Lanius",         "family": "Laniidae"},
    "Baird Sparrow":             {"genus": "Centronyx",      "family": "Passerellidae"},
    "Black throated Sparrow":    {"genus": "Amphispiza",     "family": "Passerellidae"},
    "Brewer Sparrow":            {"genus": "Spizella",       "family": "Passerellidae"},
    "Chipping Sparrow":          {"genus": "Spizella",       "family": "Passerellidae"},
    "Clay colored Sparrow":      {"genus": "Spizella",       "family": "Passerellidae"},
    "House Sparrow":             {"genus": "Passer",         "family": "Passeridae"},
    "Field Sparrow":             {"genus": "Spizella",       "family": "Passerellidae"},
    "Fox Sparrow":               {"genus": "Passerella",     "family": "Passerellidae"},
    "Grasshopper Sparrow":       {"genus": "Ammodramus",     "family": "Passerellidae"},
    "Harris Sparrow":            {"genus": "Zonotrichia",    "family": "Passerellidae"},
    "Henslow Sparrow":           {"genus": "Centronyx",      "family": "Passerellidae"},
    "Le Conte Sparrow":          {"genus": "Ammospiza",      "family": "Passerellidae"},
    "Lincoln Sparrow":           {"genus": "Melospiza",      "family": "Passerellidae"},
    "Nelson Sharp tailed Sparrow":{"genus": "Ammospiza",     "family": "Passerellidae"},
    "Savannah Sparrow":          {"genus": "Passerculus",    "family": "Passerellidae"},
    "Seaside Sparrow":           {"genus": "Ammospiza",      "family": "Passerellidae"},
    "Song Sparrow":              {"genus": "Melospiza",      "family": "Passerellidae"},
    "Tree Sparrow":              {"genus": "Spizella",       "family": "Passerellidae"},
    "Vesper Sparrow":            {"genus": "Pooecetes",      "family": "Passerellidae"},
    "White crowned Sparrow":     {"genus": "Zonotrichia",    "family": "Passerellidae"},
    "White throated Sparrow":    {"genus": "Zonotrichia",    "family": "Passerellidae"},
    "Cape Glossy Starling":      {"genus": "Lamprotornis",   "family": "Sturnidae"},
    "Bank Swallow":              {"genus": "Riparia",        "family": "Hirundinidae"},
    "Barn Swallow":              {"genus": "Hirundo",        "family": "Hirundinidae"},
    "Cliff Swallow":             {"genus": "Petrochelidon",  "family": "Hirundinidae"},
    "Tree Swallow":              {"genus": "Tachycineta",    "family": "Hirundinidae"},
    "Scarlet Tanager":           {"genus": "Piranga",        "family": "Cardinalidae"},
    "Summer Tanager":            {"genus": "Piranga",        "family": "Cardinalidae"},
    "Artic Tern":                {"genus": "Sterna",         "family": "Laridae"},
    "Black Tern":                {"genus": "Chlidonias",     "family": "Laridae"},
    "Caspian Tern":              {"genus": "Hydroprogne",    "family": "Laridae"},
    "Common Tern":               {"genus": "Sterna",         "family": "Laridae"},
    "Elegant Tern":              {"genus": "Thalasseus",     "family": "Laridae"},
    "Forsters Tern":             {"genus": "Sterna",         "family": "Laridae"},
    "Least Tern":                {"genus": "Sternula",       "family": "Laridae"},
    "Green tailed Towhee":       {"genus": "Pipilo",         "family": "Passerellidae"},
    "Brown Thrasher":            {"genus": "Toxostoma",      "family": "Mimidae"},
    "Sage Thrasher":             {"genus": "Oreoscoptes",    "family": "Mimidae"},
    "Black capped Vireo":        {"genus": "Vireo",          "family": "Vireonidae"},
    "Blue headed Vireo":         {"genus": "Vireo",          "family": "Vireonidae"},
    "Philadelphia Vireo":        {"genus": "Vireo",          "family": "Vireonidae"},
    "Red eyed Vireo":            {"genus": "Vireo",          "family": "Vireonidae"},
    "Warbling Vireo":            {"genus": "Vireo",          "family": "Vireonidae"},
    "White eyed Vireo":          {"genus": "Vireo",          "family": "Vireonidae"},
    "Yellow throated Vireo":     {"genus": "Vireo",          "family": "Vireonidae"},
    "Bay breasted Warbler":      {"genus": "Setophaga",      "family": "Parulidae"},
    "Black and white Warbler":   {"genus": "Mniotilta",      "family": "Parulidae"},
    "Black throated Blue Warbler":{"genus": "Setophaga",     "family": "Parulidae"},
    "Blue winged Warbler":       {"genus": "Vermivora",      "family": "Parulidae"},
    "Canada Warbler":            {"genus": "Cardellina",     "family": "Parulidae"},
    "Cape May Warbler":          {"genus": "Setophaga",      "family": "Parulidae"},
    "Cerulean Warbler":          {"genus": "Setophaga",      "family": "Parulidae"},
    "Chestnut sided Warbler":    {"genus": "Setophaga",      "family": "Parulidae"},
    "Golden winged Warbler":     {"genus": "Vermivora",      "family": "Parulidae"},
    "Hooded Warbler":            {"genus": "Setophaga",      "family": "Parulidae"},
    "Kentucky Warbler":          {"genus": "Geothlypis",     "family": "Parulidae"},
    "Magnolia Warbler":          {"genus": "Setophaga",      "family": "Parulidae"},
    "Mourning Warbler":          {"genus": "Geothlypis",     "family": "Parulidae"},
    "Myrtle Warbler":            {"genus": "Setophaga",      "family": "Parulidae"},
    "Nashville Warbler":         {"genus": "Leiothlypis",    "family": "Parulidae"},
    "Orange crowned Warbler":    {"genus": "Leiothlypis",    "family": "Parulidae"},
    "Palm Warbler":              {"genus": "Setophaga",      "family": "Parulidae"},
    "Pine Warbler":              {"genus": "Setophaga",      "family": "Parulidae"},
    "Prairie Warbler":           {"genus": "Setophaga",      "family": "Parulidae"},
    "Prothonotary Warbler":      {"genus": "Protonotaria",   "family": "Parulidae"},
    "Swainson Warbler":          {"genus": "Limnothlypis",   "family": "Parulidae"},
    "Tennessee Warbler":         {"genus": "Leiothlypis",    "family": "Parulidae"},
    "Wilson Warbler":            {"genus": "Cardellina",     "family": "Parulidae"},
    "Worm eating Warbler":       {"genus": "Helmitheros",    "family": "Parulidae"},
    "Yellow Warbler":            {"genus": "Setophaga",      "family": "Parulidae"},
    "Northern Waterthrush":      {"genus": "Parkesia",       "family": "Parulidae"},
    "Louisiana Waterthrush":     {"genus": "Parkesia",       "family": "Parulidae"},
    "Bohemian Waxwing":          {"genus": "Bombycilla",     "family": "Bombycillidae"},
    "Cedar Waxwing":             {"genus": "Bombycilla",     "family": "Bombycillidae"},
    "American Three toed Woodpecker":{"genus": "Picoides",   "family": "Picidae"},
    "Pileated Woodpecker":       {"genus": "Dryocopus",      "family": "Picidae"},
    "Red bellied Woodpecker":    {"genus": "Melanerpes",     "family": "Picidae"},
    "Red cockaded Woodpecker":   {"genus": "Dryobates",      "family": "Picidae"},
    "Red headed Woodpecker":     {"genus": "Melanerpes",     "family": "Picidae"},
    "Downy Woodpecker":          {"genus": "Dryobates",      "family": "Picidae"},
    "Bewick Wren":               {"genus": "Thryomanes",     "family": "Troglodytidae"},
    "Cactus Wren":               {"genus": "Campylorhynchus","family": "Troglodytidae"},
    "Carolina Wren":             {"genus": "Thryothorus",    "family": "Troglodytidae"},
    "House Wren":                {"genus": "Troglodytes",    "family": "Troglodytidae"},
    "Marsh Wren":                {"genus": "Cistothorus",    "family": "Troglodytidae"},
    "Rock Wren":                 {"genus": "Salpinctes",     "family": "Troglodytidae"},
    "Winter Wren":               {"genus": "Troglodytes",    "family": "Troglodytidae"},
    "Common Yellowthroat":       {"genus": "Geothlypis",     "family": "Parulidae"},
}

# ── Comparação ────────────────────────────────────────────────────────────────
results = []
for ep_name, ep_tax in ep_species.items():
    match_level = "Nenhuma"
    matched_cub = []

    for cub_name, cub_tax in cub_species.items():
        ep_clean  = ep_name.lower().replace(" ", "")
        cub_clean = cub_name.lower().replace(" ", "")

        if ep_clean == cub_clean or ep_name.lower() in cub_name.lower() or cub_name.lower() in ep_name.lower():
            match_level = "Exata / Nome"
            matched_cub.append(cub_name)
        elif cub_tax["genus"] == ep_tax["genus"] and match_level not in ("Exata / Nome",):
            match_level = "Mesmo gênero"
            matched_cub.append(cub_name)
        elif cub_tax["family"] == ep_tax["family"] and match_level not in ("Exata / Nome", "Mesmo gênero"):
            match_level = "Mesma família"
            matched_cub.append(cub_name)

    results.append({
        "especie_passaros": ep_name,
        "genus_ep":   ep_tax["genus"],
        "family_ep":  ep_tax["family"],
        "match_level": match_level,
        "cub_matches": ", ".join(matched_cub) if matched_cub else "—",
    })

df = pd.DataFrame(results)

# ── Imprimir tabela ───────────────────────────────────────────────────────────
order = {"Exata / Nome": 0, "Mesmo gênero": 1, "Mesma família": 2, "Nenhuma": 3}
df["sort_key"] = df["match_level"].map(order)
df = df.sort_values("sort_key").drop(columns="sort_key")

print(f"\n{'='*100}")
print(f"{'Espécie (especie_passaros)':<35} {'Gênero':<18} {'Família':<22} {'Nível':<18} {'Correspondências CUB'}")
print(f"{'='*100}")
for _, row in df.iterrows():
    cub_short = row["cub_matches"][:55] + "…" if len(row["cub_matches"]) > 55 else row["cub_matches"]
    print(f"{row['especie_passaros']:<35} {row['genus_ep']:<18} {row['family_ep']:<22} {row['match_level']:<18} {cub_short}")

# Resumo
counts = df["match_level"].value_counts()
print(f"\n{'='*50}")
print("Resumo:")
for level in ["Exata / Nome", "Mesmo gênero", "Mesma família", "Nenhuma"]:
    print(f"  {level:<20}: {counts.get(level, 0)}")
print(f"{'='*50}")

# ── Gráfico ───────────────────────────────────────────────────────────────────
colors_map = {
    "Exata / Nome":   "#4CAF50",
    "Mesmo gênero":   "#2196F3",
    "Mesma família":  "#FF9800",
    "Nenhuma":        "#F44336",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Barras por espécie
bar_colors = [colors_map[m] for m in df["match_level"]]
bars = ax1.barh(df["especie_passaros"], [1]*len(df), color=bar_colors, edgecolor="white", height=0.6)
for bar, row in zip(bars, df.itertuples()):
    label = row.cub_matches[:45] + "…" if len(row.cub_matches) > 45 else row.cub_matches
    ax1.text(0.02, bar.get_y() + bar.get_height()/2,
             f"  {row.match_level}  →  {label}", va="center", fontsize=7.5,
             color="white" if row.match_level != "Mesma família" else "white")
ax1.set_xlim(0, 1)
ax1.set_xticks([])
ax1.set_title("Correspondência por espécie", fontweight="bold", fontsize=11)
ax1.tick_params(axis="y", labelsize=8)
patches = [mpatches.Patch(color=v, label=k) for k, v in colors_map.items()]
ax1.legend(handles=patches, loc="lower right", fontsize=8)

# Pizza com contagens
level_counts = [counts.get(l, 0) for l in colors_map]
level_labels = list(colors_map.keys())
level_cols   = list(colors_map.values())
wedges, texts, autotexts = ax2.pie(
    level_counts, labels=level_labels, colors=level_cols,
    autopct="%1.0f%%", startangle=90,
    textprops={"fontsize": 10},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
ax2.set_title("Distribuição dos níveis de correspondência\n(20 espécies do especie_passaros)", fontweight="bold")

plt.suptitle("Sobreposição taxonômica: especie_passaros × CUB-200-2011",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out = "metricas_output/comparacao_taxonomica.png"
import os; os.makedirs("metricas_output", exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSalvo: {out}")
