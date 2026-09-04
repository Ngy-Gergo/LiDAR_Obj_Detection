# Előtanított baseline-ok írás nélküli auditja

## Hatókör és forrás

Az audit a `dev` worktree következő untracked anyagát kizárólag olvasta, nem
módosította:

`research/reports/20260902-pretrained-baseline-comparison/`

A riport négy sort tartalmaz: két hivatalos MMDetection3D pretrained baseline-t
(PointPillars és SECOND), valamint két projektmodellt (Voxel0075 20 epoch és
Pillar02 30 epoch). A baseline audit alatt a két hivatalos modell értendő.

## Modellek és hiteles kötések

| Modell | Hivatalos forrásconfig | Canonical run-local config SHA-256 | Checkpoint SHA-256 / méret | Run | Evaluation / benchmark |
| --- | --- | --- | --- | --- | --- |
| Official PointPillars | `pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py`, `75145966859d3f9f4bc3fcd364ee12e6b3edc221cf1cdea7c70e4274443891dc` | `e9bb043d919705925af68e4c1c26d71a0082c2c8f147c938b11ea9fb62a77a88` | `d42d15edce05b552fdd14e6412fc1d3e02207ee4799e6e3869f61bd30e730f3e` / 19 342 405 byte | `20260902T101048Z-official-pointpillars-kitti-d42d15edce05b552fdd14e64` | `20260902T112255519403Z-evaluation-ab18369ae30ea65e3de72021` / `20260902T112454193282Z-benchmark-c18cb367b928352f966fca43` |
| Official SECOND | `second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py`, `a1ca56ed4015a19c79a38ab81e74062cf26737a17611a6521013ad672e8a3c1e` | `ab804142fac97ddf2a7f786dbce7c992f0b77ff977fb4697dc74c1ddc13772b4` | `75d9305e3403e890a32d553cc414570f8321086a13edf78f295e628de8fdc851` / 22 895 421 byte | `20260902T101049Z-official-second-kitti-75d9305e3403e890a32d553c` | `20260902T112255897033Z-evaluation-e7697481d78150aa18f138ce` / `20260902T112545789371Z-benchmark-c1b07f9d5ab96b7737f0b62a` |

A forrás MMDetection3D checkout v1.4.0, commit:
`fe25f7a51d36e3702f961e198894580d83c4387b`. A run-local configok a KITTI
útvonalakhoz és a kompatibilis rotated-IoU kiértékeléshez materializált
canonical configok, ezért hashük szándékosan eltér a hivatalos forrásconfig
hashétől.

A korábban azonosított PointPillars név
`hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth`;
ez a pontos fájlnév jelenleg nem található a workspace-ben vagy a baseline
artifact-gyökérben. A jelenleg elérhető, teljes hash-sel azonosított példányok:

- `/media/ws-rtx/datastore1/official_mmdet3d_baselines/checkpoints/pointpillars_kitti_car.pth`;
- `/media/ws-rtx/datastore1/official_mmdet3d_baselines/results/20260902T101048Z-kitti-comparison-compat/canonical_inputs/pointpillars/epoch_80.pth`.

Mindkettő SHA-256 értéke `d42d15ed…30f3e`, mérete 19 342 405 byte. A régi
`configs/models/pointpillars_kitti_car.py` projektútvonal a jelenlegi HEAD-en
már nem létezik; a hiteles, ténylegesen mért run configja a fenti run
`config.py` fájlja.

## Futás- és result-integritás

- Mindkét baseline historical import, `training.status=completed`, de nincs
  saját training attemptje; a manifest helyesen `history_complete=false` és
  `resumable=false`.
- Mind a négy exact evaluation/benchmark `result.json` státusza `succeeded`.
- A library szigorú `load_run`, `load_result` és `load_comparison_report`
  ellenőrzése hibamentesen lefutott; a resultkötések run ID, canonical config
  SHA-256 és checkpoint SHA-256 szinten egyeznek.
- A két canonical checkpoint jelenleg létezik a manifest abszolút útvonalán;
  tényleges méretük és újraszámított SHA-256 értékük egyezik a manifesttel.
- A comparison azonos KITTI validation partíciót, Car osztályt, dataset
  identityt és annotation hash-eket köt össze. A KITTI release label hiánya
  explicit felmentés. Ez nem azonos tréningbudgetű abláció és nem érintetlen teszthalmazon mért
  eredmény.

## Plotok reprodukálhatósága

A `comparison.json` szigorúan betölthető, ezért a repository meglévő
`research/tools/plot.py` eszközével reprodukálható a hét PNG:

- `accuracy_3d_ap40.png`;
- `accuracy_bev_ap40.png`;
- `accuracy_vs_latency.png`;
- `latency_percentiles.png`;
- `peak_gpu_memory.png`;
- `checkpoint_size.png`;
- `comparison_table.png`.

Ezek az auditált dev riportban csak PNG-k; SVG-változat nincs. A plotok nem
kerülnek automatikusan a végleges prezentációs assetek közé, mert a baseline-ok
tréningbudgetje eltér, és még nincsenek a közös Foxglove runtime-ban validálva.

## Jelenlegi runtime-állapot

- A két baseline korábbi evaluation- és benchmark-futása bizonyítja, hogy a
  canonical config/checkpoint párok a research MMDetection3D környezetben
  modellként betölthetők voltak. Ebben az auditban nem történt új
  modellbetöltés vagy GPU-futtatás.
- A generikus, run-directory alapú `Mmdet3dDetector` kezeli a historical import
  abszolút selected-checkpoint hivatkozását, ezért ez a kutatási/legacy út
  elvileg mindkét baseline-hoz használható.
- A közös MCAP/ROS2/Foxglove út viszont a zárt `FinalistDetector` registryt
  használja. Jelenleg csak `voxel0075`, `pillar02` és
  `pillar02_multiclass` alias fogadható el; a resolver ezen felül natív runt
  követel. Emiatt sem az Official PointPillars, sem az Official SECOND nem
  indítható ma a közös Foxglove launcherrel.
- A közös runtime fix CenterPoint tartományt használ:
  `[0, -38.4, -3, 67.2, 38.4, 1]`. A PointPillars tartománya
  `[0, -39.68, -3, 69.12, 39.68, 1]`, a SECOND-é
  `[0, -40, -3, 70.4, 40, 1]`. A baseline bekötéshez ezt modellenként kell
  feloldani, nem szabad a jelenlegi fix szűrőt csendben újrahasználni.

## Output-adapter és tracker

Mindkét baseline MMDetection3D kimenete a repository által már kezelt
`pred_instances_3d` szerződést adja: 7 elemű 3D box, score és egész label. A
meglévő `_validated_prediction_arrays` adapter architektúrafüggetlen és a
kizárólag Car osztályt kezelő `label=0` esetet támogatja, ezért a detektorkimenet közös
`DetectionFrame` formára alakításához új modell-specifikus output-adapter nem
látszik szükségesnek.

Az `OnlineBoxTracker` már `DetectionFrame`-et fogyaszt, osztályazonosságra és
térbeli kapura illeszt, nem CenterPoint-feature-re. Következésképp a tracker
változtatás nélkül újrahasználható, ha a baseline detektor ugyanazt a hiteles
`DetectionFrame` szerződést adja. A registry, a modellenkénti ponttartomány, a
ROS aliasok/marker-színek és az identity-kötések bővítése ettől még szükséges.

## Következő megvalósítási terv

1. A baseline checkpointokat run-relatív, tartós artifactként csomagolni vagy
   külön, explicit historical-playback policyval feloldani; a méretet és teljes
   SHA-256 értéket zárt registryben rögzíteni.
2. A playback model-specet kiterjeszteni class nevekkel és modellenkénti
   pontfelhő-tartománnyal; megszüntetni a baseline-ra hibás fix CenterPoint
   range alkalmazását.
3. Az Official PointPillars és SECOND aliasokat végigvezetni a launcher,
   ROS2 node, marker-színek és diagnosztikai identity útvonalán.
4. Fixture-alapú CPU-tesztekkel igazolni a közös MMDetection3D output-adaptert
   és az adapter utáni változatlan tracker működését.
5. Csak ezután, külön engedélyezett fázisban egy rövid GPU smoke-frame-et és
   Foxglove dry-run/live validációt végezni, majd elkészíteni a videókat.

Ebben a fázisban kompatibilitási javítás, modellbetöltés, videó és hosszú
GPU-futtatás nem történt.
