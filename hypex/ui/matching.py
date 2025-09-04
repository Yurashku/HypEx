from __future__ import annotations

from typing import Any

from ..analyzers.matching import MatchingAnalyzer
from ..dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    StatisticRole,
    TargetRole,
)
from ..reporters.matching import MatchingDictReporter, MatchingQualityDatasetReporter
from ..utils import ID_SPLIT_SYMBOL, MATCHING_INDEXES_SPLITTER_SYMBOL
from .base import Output


class MatchingOutput(Output):
    resume: Dataset
    full_data: Dataset
    quality_results: Dataset

    def __init__(self, searching_class: type = MatchingAnalyzer):
        super().__init__(
            resume_reporter=MatchingDictReporter(searching_class),
            additional_reporters=MatchingQualityDatasetReporter(),
        )

    def _extract_full_data(self, experiment_data: ExperimentData, indexes: Dataset):
        self.indexes = Dataset(roles={}, data=experiment_data.ds.index)
        for i in range(len(indexes.columns)):
            t_indexes = indexes.iloc[:, i]
            t_indexes.index = experiment_data.ds.index
            filtered_field = indexes.drop(
                indexes[indexes[t_indexes.columns[0]] == -1], axis=0
            )
            matched_data = experiment_data.ds.loc[
                list(map(lambda x: x[0], filtered_field.get_values()))
            ].rename({col: col + f"_matched_{i}" for col in experiment_data.ds.columns})
            matched_data.index = filtered_field.index

            self.indexes = (
                t_indexes
                if self.indexes.is_empty()
                else self.indexes.add_column(t_indexes)
            )
            if hasattr(self, "full_data") and self.full_data is not None:
                self.full_data = self.full_data.append(
                    matched_data.reindex(experiment_data.ds.index), axis=1
                )
            else:
                self.full_data = experiment_data.ds.append(
                    matched_data.reindex(experiment_data.ds.index), axis=1
                )

    def _reformat_resume(self, resume: dict[str, Any]):
        reformatted_resume: dict[str, Any] = {}

        for key, value in resume.items():
            if ID_SPLIT_SYMBOL not in key:
                continue

            keys = key.split(ID_SPLIT_SYMBOL)

            if keys[0] == "indexes":
                if len(keys) > 2:
                    reformatted_resume.setdefault("indexes", {}).setdefault(
                        keys[1], {}
                    )[keys[2]] = value
                else:
                    reformatted_resume.setdefault("indexes", {})[keys[1]] = value
            else:
                l1_key = keys[0] if len(keys) < 3 else f"{keys[2]} {keys[0]}"
                reformatted_resume.setdefault(l1_key, {})[keys[1]] = value

        return reformatted_resume

    def extract(self, experiment_data: ExperimentData):
        resume = self.resume_reporter.report(experiment_data)
        reformatted_resume = self._reformat_resume(resume)
        if "indexes" in reformatted_resume.keys():
            group_indexes_id = experiment_data.ds.search_columns(GroupingRole())
            indexes = [
                Dataset.from_dict(
                    {
                        f"indexes_{group}": list(
                            map(int, values.split(MATCHING_INDEXES_SPLITTER_SYMBOL))
                        )
                    },
                    index=experiment_data.ds[
                        experiment_data.ds[group_indexes_id] == group
                    ].index,
                    roles={f"indexes_{group}": StatisticRole()},
                )
                for group, values in reformatted_resume.pop("indexes").items()
            ]
            indexes = indexes[0].append(other=indexes[1:], axis=1).sort()
        else:
            indexes = Dataset.from_dict(
                {
                    "indexes": list(
                        map(
                            int,
                            resume["indexes"].split(MATCHING_INDEXES_SPLITTER_SYMBOL),
                        )
                    )
                },
                roles={"indexes": AdditionalMatchingRole()},
            )

        outcome = experiment_data.field_search(TargetRole())[0]
        reformatted_resume["outcome"] = {
            key: outcome
            for key in reformatted_resume[next(iter(reformatted_resume.keys()))].keys()
        }

        self.resume = Dataset.from_dict(
            reformatted_resume,
            roles={
                column: StatisticRole() for column in list(reformatted_resume.keys())
            },
        )
        self._extract_full_data(
            experiment_data,
            indexes,
        )
        self.resume = round(self.resume, 2)

        self.quality_results = self.additional_reporters.report(experiment_data)
