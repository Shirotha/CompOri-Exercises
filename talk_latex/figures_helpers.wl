RealNumberQ = Except[_Complex, _?NumberQ];

pattern[c1 : _?ColorQ, c2 : _?ColorQ, ratio : _?RealNumberQ] :=
    PatternFilling[
        Graphics[{
            c1,
            Triangle@{
                {0, 0},
                {1, 0},
                {0, ratio}
            },
            c2,
            Triangle@{
                {1, 0},
                {1, ratio},
                {0, ratio}
            }
        },
        Method -> {Automatic, "TransparentPolygonMesg" -> True}
        ]
    ];

SharedColorDirective[segments : _?IntegerQ, points : _?IntegerQ, stride : ?IntegerQ, i : _?IntegerQ, cf : _Function, ratio : _?RealNumberQ] :=
    With[{
        leftSegment = Clip[Floor[(i - 1) / 2], {0, segments - 1}],
        rightSegment = Clip[Floor[i / 2], {0, segments - 1}]
    },
    If[leftSegment === rightSegment,
        cf[leftSegment, i - leftSegment stride]
    ,
        pattern[
            cf[leftSegment, i - leftSegment stride],
            cf[rightSegment, i - rightSegment stride],
            ratio
        ]
    ]
    ];

(* NOTE: n, j are zero based *)
Clear@ControlPointList;
Options@ControlPointList = {
    ColorFunction -> {n, j} |-> ColorData[97, n],
    LabelFunction -> {n, j} |-> "", (* TODO: how to write both labels in shared points *)
    PatternRatio -> 1,
    AspectRatio -> 1
};
ControlPointList[segments : _?IntegerQ, points : _?IntegerQ, stride : _?IntegerQ, OptionsPattern[]]] :=
    With[{
        totalPoints = segments stride + (points - stride),
        colors = OptionValue[ColorFunction],
        labels = OptionValue[LabelFunction],
        pRatio = OptionValue[PatternRatio],
        aRatio = OptionValue[AspectRatio]
    },
    Graphics@Flatten@Table[
        {
            SharedColorDirective[segments, points, stride, i, colors, pRatio],
            Rectangle@{{i, 0}, {i + 1, totalPoints aRatio}}]
            (* TODO: draw labels here *)
        ]
    , {i, 0, totalPoints - 1}]
    ];

Clear@SplineExample;
Options@SplineExample = {
    ColorFunction -> {n, j} |-> ColorData[97, n],
    PatternRatio -> 1
}
SplineExample[order : _?IntegerQ, controlPoints : {{_?RealNumberQ, _?RealNumberQ}..}, OptionsPattern[]] :=
    With[{
        colors = OptionValue[ColorFunction],
        pRatio = OptionValue[PatternRatio]
    },
    Graphics@{
        Thick,
        ColorData[97, 1],
        BezierCurve[controlPoints, SplineDegree -> order],
        Black,
        Dashed,
        Line@controlPoints,
        Sequence@@Flatten@Table[
            {
                SharedColorDirective[(Length@controlPoints - 1) / order, order + 1, order, i, colors, pRatio],
                Point[controlPoints[[i + 1]]]
            }
        , {i, 0, Length@controlPoints - 1}
        ]
    }
    ];