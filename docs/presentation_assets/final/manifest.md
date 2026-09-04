# Prezentációs asset- és evidence-manifest

Ez a csomag csak már elkészült, változatlan `result.json` és feloldott
összehasonlító JSON-forrásokból készül. A renderelő nem indít tanítást,
kiértékelést, benchmarkot vagy modellbetöltést. A
`research/evaluations/summary.csv` és a `research/benchmarks/summary.csv` nem
adatforrás.

Az evidence manifest és a magyarázó szöveg magyar. A négy végleges ábra belső,
felhasználó által olvasott címei, tengelyfeliratai, jelmagyarázatai, mátrix- és
táblázatfejlécei, annotációi és megjegyzései viszont tudatosan angol nyelvűek,
hogy a prezentáció nemzetközi közönség számára is közvetlenül olvasható legyen.

## Forrás és megőrzési állapot

Az ábrák a lent megnevezett, változatlan immutable evaluation- és benchmark-
forrásokhoz kötött végleges SVG/PNG assetek. A `summary.csv` állományok nem
adatforrások.

## Megtartott ábrák

### `hatmodell_pareto_3d_ap40_p95`

- Szakmai kérdés: milyen pontosság–késleltetés kompromisszumot ad a hat, csak
  Car osztályt kezelő, 20 epochos modell, és melyek nem domináltak?
- Következtetés: a Pillar02 adja a legkisebb késleltetést, a Voxel0075 a
  legnagyobb közepes 3D AP40 értéket; a DCN változatok nem kerülnek a
  Pareto-határra.
- Forrásfájl:
  `research/reports/20260827-six-model-20epoch/fresh-end-to-end-p95.json`.
- Bemutatott metrika: Car 3D AP40 strict, közepes nehézség; végponttól
  végpontig mért p95 késleltetés.
- Összehasonlíthatósági korlát: csak a hat azonos kampányú, kizárólag Car
  osztályt kezelő, 20 epochos futás rangsorolható együtt; egyetlen seed és KITTI validation
  eredmény.

### `hatmodell_p95_meresi_scope`

- Szakmai kérdés: mekkora a különbség a csak predikciót és a teljes
  adatbetöltés-plusz-predikció utat mérő p95 között?
- Következtetés: mind a hat modell teljesíti az 50 ms-os 20 Hz követelményt;
  a két mérési tartomány különbsége minden modellnél kicsi a modellválasztásból eredő
  késleltetéskülönbséghez képest.
- Forrásfájlok:
  `research/reports/20260827-six-model-20epoch/fresh-prediction-p95.json` és
  `research/reports/20260827-six-model-20epoch/fresh-end-to-end-p95.json`.
- Bemutatott metrika: prediction p95 és end-to-end p95, milliszekundumban.
- Összehasonlíthatósági korlát: a mérési tartományok két külön, feloldott riportból
  származnak, de modellről modellre azonos checkpointkötésűek; a p95 értékek
  különbsége nem eseményszintű időfelbontás.

### `finalistak_eredmenymatrix`

- Szakmai kérdés: ugyanazon architektúrán belül hogyan változik a teljes
  pontossági profil és a p95 késleltetés 20-ról 30 epochra?
- Következtetés: a Voxel0075 30 epochnál minden bemutatott AP40-szeletben és
  mindkét p95 mérési tartományban romlik; a Pillar02 mind a hat AP40-szeletben javul,
  miközben a p95 csak kismértékben nő.
- Forrásfájl:
  `research/reports/20260902-finalists-duration30/paired-end-to-end-p95.json`.
- Bemutatott metrika: Car 3D és BEV AP40 strict könnyű/közepes/nehéz;
  prediction és end-to-end p95.
- Összehasonlíthatósági korlát: csak a Pillar02 és Voxel0075 architektúrán
  belüli párosított 20/30 epochos következtetés érvényes; az eredmények
  egyetlen seeddel készült KITTI validation mérések.

### `pillar02_tobbosztalyos_ap40_matrix`

- Szakmai kérdés: milyen az epoch 55 checkpoint Car, Pedestrian és Cyclist
  teljesítménye 3D-ben és BEV-ben, nehézségi szintenként?
- Következtetés: a Car a legerősebb osztály; a Pedestrian a legnehezebb, míg
  a Cyclist 3D teljesítménye erősen csökken a könnyűről közepes/nehéz szintre.
  A 14,77 ms-os teljes p95 alapján a kísérlet teljesíti a 20 Hz követelményt.
- Forrásfájlok:
  `research/runs/20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac/evaluation/20260902T195616343969Z-evaluation-513fc499ed4e49415ee237c2/result.json`
  és
  `research/runs/20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac/benchmark/20260902T195651982215Z-benchmark-0001486e79f9c7dd48f28e10/result.json`.
- Bemutatott metrika: Car, Pedestrian és Cyclist 3D/BEV AP40 strict,
  könnyű/közepes/nehéz; end-to-end p95 kiegészítő evidence-ként.
- Összehasonlíthatósági korlát: ez külön Pillar02 többosztályos,
  60 epochos kísérlet, amelynek kijelölt checkpointja az 55. epochból származik.
  Nem rangsorolható közvetlenül a hat 20 epochos, kizárólag Car osztályt
  kezelő modellel.

## Multiclass artifactkötés

- Tartós runs-root:
  `/home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs`.
- Run ID: `20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac`.
- Config SHA-256:
  `b31131058eb44367a6d7daa7a3ee0620d41cf3324fdfd701f1f41d402a50231d`.
- Kijelölt checkpoint:
  `training/best_Kitti metric_pred_instances_3d_KITTI_Car_3D_AP40_moderate_strict_epoch_55.pth`.
- Checkpointméret: 32 452 134 byte.
- Checkpoint SHA-256:
  `cf62f3c99ce8ebdbb96eaa467cc44c3bde0a152aa25470ce1cfd11b8ac7c7427`.

## Tudatosan kihagyott ábrák

- Külön hatmodelles 3D és BEV oszlopdiagram: a 3D rangsort a Pareto-ábra már
  tartalmazza, a BEV rangsor pedig nem változtatja meg a modellválasztási
  következtetést; külön ábraként ismétlés lenne.
- Egyosztályos keveredési mátrix: a kizárólag Car osztályt kezelő finalistáknál nem adna többet a
  meglévő AP40- és késleltetésevidence-nél.
- Multiclass confusion matrix és PR-görbe: az evaluation `result.json` csak
  126 aggregált metrikát tárol; nincs benne per-frame predikció, score-sorozat,
  párosítás vagy ground truth. Ezek AP40-ből nem számíthatók vissza.
- Kvalitatív detektálási diagram: meglévő aggregált JSON-ból nem reprodukálható
  hitelesen. Annotált képkocka vagy külön Foxglove-videó lesz a megfelelő
  következő fázis.
- Előtanított baseline-ábra: ebben a fázisban a baseline-anyag csak írás nélküli
  audit tárgya; Foxglove-integráció és presentation-ready ábra külön feladat.
