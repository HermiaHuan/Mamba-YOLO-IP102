from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "pest102"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")

DEFAULT_CLASS_NAMES = [
    "rice leaf roller",
    "rice leaf caterpillar",
    "paddy stem maggot",
    "asiatic rice borer",
    "yellow rice borer",
    "rice gall midge",
    "Rice Stemfly",
    "brown plant hopper",
    "white backed plant hopper",
    "small brown plant hopper",
    "rice water weevil",
    "rice leafhopper",
    "grain spreader thrips",
    "rice shell pest",
    "grub",
    "mole cricket",
    "wireworm",
    "white margined moth",
    "black cutworm",
    "large cutworm",
    "yellow cutworm",
    "red spider",
    "corn borer",
    "army worm",
    "aphids",
    "Potosiabre vitarsis",
    "peach borer",
    "english grain aphid",
    "green bug",
    "bird cherry-oataphid",
    "wheat blossom midge",
    "penthaleus major",
    "longlegged spider mite",
    "wheat phloeothrips",
    "wheat sawfly",
    "cerodonta denticornis",
    "beet fly",
    "flea beetle",
    "cabbage army worm",
    "beet army worm",
    "Beet spot flies",
    "meadow moth",
    "beet weevil",
    "sericaorient alismots chulsky",
    "alfalfa weevil",
    "flax budworm",
    "alfalfa plant bug",
    "tarnished plant bug",
    "Locustoidea",
    "lytta polita",
    "legume blister beetle",
    "blister beetle",
    "therioaphis maculata Buckton",
    "odontothrips loti",
    "Thrips",
    "alfalfa seed chalcid",
    "Pieris canidia",
    "Apolygus lucorum",
    "Limacodidae",
    "Viteus vitifoliae",
    "Colomerus vitis",
    "Brevipoalpus lewisi McGregor",
    "oides decempunctata",
    "Polyphagotars onemus latus",
    "Pseudococcus comstocki Kuwana",
    "parathrene regalis",
    "Ampelophaga",
    "Lycorma delicatula",
    "Xylotrechus",
    "Cicadella viridis",
    "Miridae",
    "Trialeurodes vaporariorum",
    "Erythroneura apicalis",
    "Papilio xuthus",
    "Panonchus citri McGregor",
    "Phyllocoptes oleiverus ashmead",
    "Icerya purchasi Maskell",
    "Unaspis yanonensis",
    "Ceroplastes rubens",
    "Chrysomphalus aonidum",
    "Parlatoria zizyphus Lucus",
    "Nipaecoccus vastalor",
    "Aleurocanthus spiniferus",
    "Tetradacus c Bactrocera minax",
    "Dacus dorsalis(Hendel)",
    "Bactrocera tsuneonis",
    "Prodenia litura",
    "Adristyrannus",
    "Phyllocnistis citrella Stainton",
    "Toxoptera citricidus",
    "Toxoptera aurantii",
    "Aphis citricola Vander Goot",
    "Scirtothrips dorsalis Hood",
    "Dasineura sp",
    "Lawana imitata Melichar",
    "Salurnis marginella Guerr",
    "Deporaus marginatus Pascoe",
    "Chlumetia transversa",
    "Mango flat beak leafhopper",
    "Rhytidodera bowrinii white",
    "Sternochetus frigidus",
    "Cicadellidae",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert the official IP102 archive into this Mamba-YOLO layout.")
    parser.add_argument("--zip", dest="zip_path", type=Path, default=None, help="path to archive.zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output dataset directory")
    parser.add_argument("--overwrite", action="store_true", help="replace the output directory if it already exists")
    parser.add_argument(
        "--mode",
        choices=("auto", "classification", "voc"),
        default="auto",
        help="archive format to convert",
    )
    return parser.parse_args()


def resolve_archive(zip_path: Path | None) -> Path:
    if zip_path:
        resolved = zip_path.expanduser()
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        if not resolved.exists():
            raise FileNotFoundError(f"Archive not found: {resolved}")
        return resolved

    for name in ("archive.zip", "archiev.zip"):
        candidate = PROJECT_ROOT / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Place archive.zip in the project root or pass --zip /path/to/archive.zip")


def clean_class_name(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*\d+\s+", "", line)
    return re.sub(r"\s+", " ", line).strip()


def load_class_names(zf: zipfile.ZipFile) -> list[str]:
    class_files = [name for name in zf.namelist() if PurePosixPath(name).name.lower() == "classes.txt"]
    if class_files:
        selected = sorted(class_files, key=lambda item: (len(PurePosixPath(item).parts), item))[0]
        lines = zf.read(selected).decode("utf-8", errors="ignore").splitlines()
        names = [clean_class_name(line) for line in lines if clean_class_name(line)]
        if names:
            return names
    return DEFAULT_CLASS_NAMES


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_metadata(output: Path, class_names: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    lines = [
        f"path: {output.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {yaml_quote(name)}" for idx, name in enumerate(class_names))
    (output / "pest102.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to rebuild it.")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_zip_member(zf: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def classification_image_members(zf: zipfile.ZipFile) -> list[tuple[str, str, str, Path]]:
    members: list[tuple[str, str, str, Path]] = []
    for member in zf.namelist():
        path = PurePosixPath(member)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.parts
        if "classification" not in parts:
            continue
        root_index = parts.index("classification")
        if len(parts) <= root_index + 3:
            continue
        split, class_dir = parts[root_index + 1], parts[root_index + 2]
        if split not in SPLITS:
            continue
        relative = Path(*parts[root_index + 1 :])
        members.append((member, split, class_dir, relative))
    return members


def has_classification_archive(zf: zipfile.ZipFile) -> bool:
    return bool(classification_image_members(zf))


def class_id_from_dir(class_dir: str, class_names: list[str]) -> int:
    if class_dir.isdigit():
        class_id = int(class_dir)
    else:
        lookup = {name.lower(): idx for idx, name in enumerate(class_names)}
        class_id = lookup.get(class_dir.lower(), -1)
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(f"Unsupported class directory '{class_dir}'. Expected 0-{len(class_names) - 1}.")
    return class_id


def convert_classification_archive(zf: zipfile.ZipFile, output: Path, class_names: list[str]) -> dict[str, int]:
    counts = defaultdict(int)
    members = classification_image_members(zf)
    if not members:
        raise RuntimeError("No classification/train|val|test images found in archive.")

    for member, split, class_dir, relative in members:
        class_id = class_id_from_dir(class_dir, class_names)
        image_path = output / "images" / relative
        label_path = (output / "labels" / relative).with_suffix(".txt")
        copy_zip_member(zf, member, image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        counts[split] += 1

    return dict(counts)


def has_voc_archive(zf: zipfile.ZipFile) -> bool:
    return any("/Annotations/" in f"/{name}" and name.lower().endswith(".xml") for name in zf.namelist())


def find_voc_root(zf: zipfile.ZipFile) -> PurePosixPath:
    annotation_members = [
        PurePosixPath(name)
        for name in zf.namelist()
        if "/Annotations/" in f"/{name}" and name.lower().endswith(".xml")
    ]
    if not annotation_members:
        raise RuntimeError("No VOC Annotations/*.xml files found in archive.")
    return annotation_members[0].parent.parent


def read_text(zf: zipfile.ZipFile, member: str) -> str:
    return zf.read(member).decode("utf-8", errors="ignore")


def parse_xml(zf: zipfile.ZipFile, member: str) -> ET.Element:
    text = read_text(zf, member).strip()
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        match = re.search(r"<annotation>.*</annotation>", text, re.DOTALL)
        if not match:
            raise
        return ET.fromstring(match.group(0))


def list_under(zf: zipfile.ZipFile, folder: PurePosixPath) -> list[str]:
    prefix = folder.as_posix().rstrip("/") + "/"
    return [name for name in zf.namelist() if name.startswith(prefix)]


def load_split_ids(zf: zipfile.ZipFile, voc_root: PurePosixPath, split_name: str) -> list[str]:
    candidates = [
        (voc_root / "ImageSets" / "Main" / f"{split_name}.txt").as_posix(),
        (voc_root / f"{split_name}.txt").as_posix(),
        f"{split_name}.txt",
    ]
    for candidate in candidates:
        if candidate in zf.namelist():
            return [PurePosixPath(line.strip()).stem for line in read_text(zf, candidate).splitlines() if line.strip()]
    return []


def collect_voc_class_tokens(zf: zipfile.ZipFile, annotations: dict[str, str]) -> list[str]:
    tokens: list[str] = []
    for member in annotations.values():
        root = parse_xml(zf, member)
        for obj in root.findall("object"):
            token = (obj.findtext("name") or "").strip()
            if token:
                tokens.append(token)
    return tokens


def should_shift_one_based_ids(tokens: list[str], class_count: int) -> bool:
    numeric = [int(token) for token in tokens if token.isdigit()]
    if not numeric or 0 in numeric:
        return False
    return min(numeric) >= 1 and max(numeric) <= class_count


def resolve_voc_class_id(raw_name: str, class_names: list[str], one_based: bool) -> int:
    raw_name = raw_name.strip()
    if raw_name.isdigit():
        class_id = int(raw_name) - 1 if one_based else int(raw_name)
    else:
        lookup = {name.lower(): idx for idx, name in enumerate(class_names)}
        class_id = lookup.get(raw_name.lower(), -1)
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(f"Unsupported VOC class '{raw_name}'.")
    return class_id


def find_image_size(root: ET.Element) -> tuple[float, float]:
    size = root.find("size")
    if size is None:
        raise ValueError("VOC XML has no <size> block.")
    width = float(size.findtext("width") or 0)
    height = float(size.findtext("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("VOC XML image size is invalid.")
    return width, height


def convert_xml_boxes(root: ET.Element, class_names: list[str], one_based: bool) -> list[str]:
    img_w, img_h = find_image_size(root)
    rows: list[str] = []
    for obj in root.findall("object"):
        class_id = resolve_voc_class_id(obj.findtext("name") or "", class_names, one_based)
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = max(0.0, float(bbox.findtext("xmin") or 0))
        ymin = max(0.0, float(bbox.findtext("ymin") or 0))
        xmax = min(img_w, float(bbox.findtext("xmax") or 0))
        ymax = min(img_h, float(bbox.findtext("ymax") or 0))
        if xmax <= xmin or ymax <= ymin:
            continue
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        rows.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return rows


def build_voc_member_maps(zf: zipfile.ZipFile, voc_root: PurePosixPath) -> tuple[dict[str, str], dict[str, str]]:
    annotations = {
        PurePosixPath(member).stem: member
        for member in list_under(zf, voc_root / "Annotations")
        if member.lower().endswith(".xml")
    }
    images = {
        PurePosixPath(member).stem: member
        for member in list_under(zf, voc_root / "JPEGImages")
        if PurePosixPath(member).suffix.lower() in IMAGE_SUFFIXES
    }
    return annotations, images


def convert_voc_archive(zf: zipfile.ZipFile, output: Path, class_names: list[str]) -> dict[str, int]:
    voc_root = find_voc_root(zf)
    annotations, images = build_voc_member_maps(zf, voc_root)
    if not annotations or not images:
        raise RuntimeError("VOC archive must contain Annotations/*.xml and JPEGImages/*.")

    train_ids = load_split_ids(zf, voc_root, "train") or load_split_ids(zf, voc_root, "trainval")
    val_ids = load_split_ids(zf, voc_root, "val") or load_split_ids(zf, voc_root, "test")
    test_ids = load_split_ids(zf, voc_root, "test") or val_ids
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    if not train_ids or not val_ids:
        raise RuntimeError("VOC archive must contain train/trainval and val/test split files.")

    one_based = should_shift_one_based_ids(collect_voc_class_tokens(zf, annotations), len(class_names))
    counts = defaultdict(int)

    for split, ids in split_ids.items():
        for image_id in ids:
            stem = PurePosixPath(image_id).stem
            annotation_member = annotations.get(stem)
            image_member = images.get(stem)
            if not annotation_member or not image_member:
                continue

            root = parse_xml(zf, annotation_member)
            labels = convert_xml_boxes(root, class_names, one_based)
            if not labels:
                continue

            image_name = PurePosixPath(image_member).name
            image_path = output / "images" / split / image_name
            label_path = output / "labels" / split / f"{stem}.txt"
            copy_zip_member(zf, image_member, image_path)
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
            counts[split] += 1

    return dict(counts)


def choose_mode(zf: zipfile.ZipFile, requested: str) -> str:
    if requested != "auto":
        return requested
    if has_voc_archive(zf):
        return "voc"
    if has_classification_archive(zf):
        return "classification"
    raise RuntimeError("Could not detect IP102 archive format.")


def main() -> None:
    args = parse_args()
    archive_path = resolve_archive(args.zip_path)
    output = args.output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    with zipfile.ZipFile(archive_path) as zf:
        mode = choose_mode(zf, args.mode)
        class_names = load_class_names(zf)
        prepare_output(output, args.overwrite)
        if mode == "classification":
            counts = convert_classification_archive(zf, output, class_names)
            label_note = "whole-image pseudo boxes"
        else:
            counts = convert_voc_archive(zf, output, class_names)
            label_note = "VOC bounding boxes"
        write_metadata(output, class_names)

    print(f"Converted IP102 archive: {archive_path}")
    print(f"Mode: {mode} ({label_note})")
    print(f"Output: {output.resolve()}")
    print(f"Classes: {len(class_names)}")
    for split in SPLITS:
        print(f"{split}: {counts.get(split, 0)} images / {counts.get(split, 0)} labels")
    print(f"Dataset yaml: {(output / 'pest102.yaml').resolve()}")


if __name__ == "__main__":
    main()
