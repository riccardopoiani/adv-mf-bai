from algorithms.bai import BaiAlgorithm, BAIConfig
from algorithms.gradient import BAIGradientLearner, MFBAIGradientLearner
from algorithms.lucb import LUCBExploreA, LUCBExploreB
from algorithms.successive_elimination import IISE


class BAIFactory:
    algo_map = {IISE.NAME: IISE,
                BAIGradientLearner.NAME: BAIGradientLearner,
                MFBAIGradientLearner.NAME: MFBAIGradientLearner,
                LUCBExploreA.NAME: LUCBExploreA,
                LUCBExploreB.NAME: LUCBExploreB
                }

    @staticmethod
    def get_algo(algo_name: str,
                 bai_cfg: BAIConfig
                 ) -> BaiAlgorithm:
        assert algo_name in list(BAIFactory.algo_map.keys()), f"Algorithm {algo_name} is not implemented."

        return BAIFactory.algo_map[algo_name](bai_cfg)
