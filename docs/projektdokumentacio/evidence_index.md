# Saját projektbizonyítékok indexe

Ez az index elkülöníti a repositoryban mért saját evidence-t a
[`references.bib`](references.bib) külső szakirodalmi és hivatalos
dokumentációs forrásaitól. A `summary.csv` állományok nem szerepelnek itt,
mert nem a dokumentáció hivatalos eredményforrásai.

| Azonosító | Tartalom és felhasználás | Elsődleges, immutable vagy feloldott forrás |
| --- | --- | --- |
| E1 | Hat Car-only CenterPoint modell 20 epochos, 3D AP40 és end-to-end p95 összevetése | [`../../research/reports/20260827-six-model-20epoch/fresh-end-to-end-p95.json`](../../research/reports/20260827-six-model-20epoch/fresh-end-to-end-p95.json) |
| E2 | Ugyanez prediction p95 tartományban | [`../../research/reports/20260827-six-model-20epoch/fresh-prediction-p95.json`](../../research/reports/20260827-six-model-20epoch/fresh-prediction-p95.json) |
| E3 | Voxel0075 és Pillar02 páros 20/30 epochos vizsgálata | [`../../research/reports/20260902-finalists-duration30/paired-end-to-end-p95.json`](../../research/reports/20260902-finalists-duration30/paired-end-to-end-p95.json) |
| E4 | Pillar02 multiclass evaluation; Car/Pedestrian/Cyclist AP40 | [`../../research/runs/20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac/evaluation/20260902T195616343969Z-evaluation-513fc499ed4e49415ee237c2/result.json`](../../research/runs/20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac/evaluation/20260902T195616343969Z-evaluation-513fc499ed4e49415ee237c2/result.json) |
| E5 | Pillar02 multiclass benchmark; p95, memória, hardver és módszertan | [`../../research/runs/20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac/benchmark/20260902T195651982215Z-benchmark-0001486e79f9c7dd48f28e10/result.json`](../../research/runs/20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac/benchmark/20260902T195651982215Z-benchmark-0001486e79f9c7dd48f28e10/result.json) |
| E6 | Multiclass run-azonosság, dataset- és checkpointkötés | [`../../research/runs/20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac/manifest.json`](../../research/runs/20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac/manifest.json) |
| E7 | Official PointPillars, Official SECOND és két projektfinalista azonos adatú inference-összevetése | [`../../research/reports/20260902-pretrained-baseline-comparison/comparison.json`](../../research/reports/20260902-pretrained-baseline-comparison/comparison.json) |
| E8 | Final presentation-ábrák forrásai, döntési értéke és reprodukciója | [`../presentation_assets/final/manifest.md`](../presentation_assets/final/manifest.md) |
| E9 | Pretrained baseline-ok config-, checkpoint- és runtime-auditja | [`../presentation_assets/final/pretrained_baseline_audit.md`](../presentation_assets/final/pretrained_baseline_audit.md) |
| E10 | A védelem alá helyezett demóválasztás, acceptance és korlátok | [`../presentation_handoff.md`](../presentation_handoff.md) |
| E11 | ROS2/Foxglove indítás, topicok, queue-szabály és acceptance | [`../tracked_foxglove_demo.md`](../tracked_foxglove_demo.md) |

## Run-owned eredmények

Az E1 és E3 comparison JSON-ok minden sorhoz konkrét evaluation- és benchmark
result ID-t, config SHA-256-ot és checkpoint SHA-256-ot kötnek. A részletes
run-owned fájlok közvetlen elérési útjai:

| Modell | Run ID | Evaluation result ID | Benchmark result ID |
| --- | --- | --- | --- |
| pillar02, 20 epoch | `20260827T092033Z-pillar02-3367910930525d0c12ddc346` | `20260901T093536573158Z-evaluation-e06d68a8b48bd9b854eaf052` | `20260901T103846232075Z-benchmark-3e3b0fb771aaacf47599a66d` |
| pillar02-dcn, 20 epoch | `20260827T092042Z-pillar02-dcn-17f8f3e630e66376d794960d` | `20260901T093605946058Z-evaluation-ce61985392addbb91ac07485` | `20260901T103919909199Z-benchmark-64f38a33a633ba14d1c6700e` |
| voxel0075, 20 epoch | `20260827T092043Z-voxel0075-e583a40f435e3071e0cbd6fc` | `20260901T093742934083Z-evaluation-1da12da1dcea037b32fcd47a` | `20260901T104014393729Z-benchmark-8788ddfdc671d76144efcac4` |
| voxel0075-dcn, 20 epoch | `20260827T092044Z-voxel0075-dcn-04444a4b945b155c3942a099` | `20260901T093838267394Z-evaluation-6a4dd2e8acccca1273132a53` | `20260901T104109572525Z-benchmark-f40cac7046c76d3d982c2ccb` |
| voxel01, 20 epoch | `20260827T092045Z-voxel01-40cd6123fa5b4cdee59306ed` | `20260901T093858828163Z-evaluation-526969cdb8c5cd6b85a8ccf8` | `20260901T104201117381Z-benchmark-e6188523989102d73e2cd36a` |
| voxel01-dcn, 20 epoch | `20260827T092046Z-voxel01-dcn-bc6db6f99a45864e67165106` | `20260901T094034238698Z-evaluation-4ee16e9ad16608a9fffc4718` | `20260901T104253301390Z-benchmark-7acb9dc627ededb2ead2734b` |
| voxel0075, 30 epoch | `20260901T195406Z-voxel0075-duration30-2ad23907052ef315ba8f8675` | `20260902T082628281593Z-evaluation-2fdaa37945bd2675c557dfef` | `20260902T083009602325Z-benchmark-f58a76e9462024acdb0df19e` |
| pillar02, 30 epoch | `20260901T195416Z-pillar02-duration30-2720f37cf422c4e55bafd0a6` | `20260902T082842231699Z-evaluation-f67bfeac549a966c8ff58b73` | `20260902T083106887597Z-benchmark-943a7b5978dd17570bd393c9` |
| pillar02 multiclass, 60 epoch | `20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac` | `20260902T195616343969Z-evaluation-513fc499ed4e49415ee237c2` | `20260902T195651982215Z-benchmark-0001486e79f9c7dd48f28e10` |
