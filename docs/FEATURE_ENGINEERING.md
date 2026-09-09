# Feature engineering métier

`FeatureEngineeringPipeline` transforme les mesures brutes en variables
explicables pour la classification du risque métabolique.

| Famille | Exemples | Information représentée |
|---|---|---|
| Démographie | `age_group`, `age_squared` | Effet non linéaire de l'âge |
| Glycémie | `homa_ir`, `glycemic_excursion_ratio` | Résistance à l'insuline et réponse postprandiale |
| Lipides | `non_hdl_cholesterol`, `triglyceride_hdl_ratio` | Charge athérogène et équilibre lipidique |
| Pression artérielle | `pulse_pressure`, `mean_arterial_pressure` | Composantes complémentaires de la pression |
| Métabolisme | `tyg_index`, `tyg_bmi_index` | Interaction triglycérides, glycémie et adiposité |
| Antécédents | `medical_history_burden` | Accumulation d'antécédents disponibles |
| Comportement | `activity_deficit_ratio`, `sleep_deviation_hours` | Écart aux repères d'activité et de sommeil |
| Synthèse | `cardiometabolic_burden`, `lifestyle_score` | Accumulation de facteurs de risque modifiables |

Les colonnes `diabetes_risk_score` et `diabetes_stage` sont supprimées avant
l'entraînement, car elles encodent directement ou indirectement la cible.
`diagnosed_diabetes` est uniquement préservée comme cible quand elle est passée
avec `target_column`; elle n'entre jamais dans les transformations.

Ces variables servent à la modélisation et à l'analyse. Elles ne constituent
pas un diagnostic clinique individuel.
