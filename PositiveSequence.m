(* ----------------------------------------------------------------------------*)
(* Copyright (C) 2022 Philipp Nuspl, philipp.nuspl@jku.at                      *)
(*                                                                             *)
(* This program is free software: you can redistribute it and/or modify it     *)
(* under the terms of the GNU General Public License as published by the Free  *)
(* Software Foundation, either version 3 of the License, or (at your option)   *)
(* any later version.                                                          *)
(*                                                                             *)
(* This program is distributed in the hope that it will be useful, but WITHOUT *)
(* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *)
(* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *)
(* more details.                                                               *)
(*                                                                             *)
(* You should have received a copy of the GNU General Public License along     *)
(* with this program. If not, see <https://www.gnu.org/licenses/>.             *)
(* ----------------------------------------------------------------------------*)

(* Tests run through until Mathematica 10 (but not in Mathematica 9)           *)

BeginPackage["PositiveSequence`", {"RISC`GeneratingFunctions`"}] 

CFiniteClosedForm::usage="CFiniteClosedForm[seq, n] returns a list of pairs " <>
  "(poly, ev) describing the closed form of seq, i.e., the sum over " <>
  "poly(n)*ev^n is seq. \n\n" <>
  "seq... a C-finite sequence as an RE[...] object \n" <>
  "n... the variable of the polynomial factors of the closed form"
SequenceFromExpression ::usage="SequenceFromExpression [expr, w[n]] returns " <>
  "a C-finite sequence " <>
  "representing the given expression. The given expression has to be a " <>
  "symbolic expression in one variable. Then, 100 terms of the sequence " <>
  "are computed and a C-finite recurrence of order at most 50 is guessed.\n\n" <>
  "expr... a symbolic expression in one variable \n" <>
  "w[n]... the name of the sequence created, i.e., the sequence has the form" <>
  "RE[{...}, w[n]]"
KPAlgorithm1::usage="KPAlgorithm1[seq] uses Algorithm 1 from " <> 
  "[Kauers, Pillwein 2010] " <>
  "to check whether the sequence is positive. Returns True if the sequence " <>
  "could be proven to be positive (i.e., every term of the sequence is " <>
  "positive) and False if a negative term was found \n\n" <>
  "seq... a D-finite sequence as an RE[...] object \n\n" <>
  "Options: \n" <>
  "Strict -> if True, positivity is checked, if False, non-negativity " <>
  " is checked (default: True). \n" <>
  "Eventual -> if True, an index is returned from which on the sequence is " <>
  "guaranteed to be positive/non-negative (default: False) \n" <>
  "Verbose -> displays information during the execution of the algorithm " <>
  "(default: False)"
KPAlgorithm2::usage="KPAlgorithm2[seq] uses uses Algorithm 2 from " <> 
  "[Kauers, Pillwein 2010] " <>
  "to check whether the sequence is positive. Returns True if the sequence " <>
  "could be proven to be positive (i.e., every term of the sequence is " <>
  "positive) and False if a negative term was found \n\n" <>
  "seq... a D-finite sequence as an RE[...] object \n\n" <>
  "Options: \n" <>
  "Strict -> if True, positivity is checked, if False, non-negativity " <>
  " is checked (default: True). \n" <>
  "Eventual -> if True, an index is returned from which on the sequence is " <>
  "guaranteed to be positive/non-negative (default: False) \n" <>
  "Verbose -> displays information during the execution of the algorithm " <>
  "(default: False)"
IsDegenerate::usage="IsDegenerate[seq] returns True if the given C-finite " <>
  "sequence is degenerate, i.e., if the ratio of two distinct eigenvalues " <>
  "is a root of unity. Otherwise, i.e., if the sequence is non-degenerate, " <>
  "it returns False.\n\n" <>
  "seq... a C-finite sequence as an RE[...] object"
DecomposeDegenerate::usage="DecomposeDegenerate[seq] returns a list of " <>
  "sequences {seq1,...,seqk} such that seq is the interlacing of " <>
  "seq1,...,seqk and each of these sequences is non-degenerate or zero.\n\n" <>
  "seq... a C-finite sequence as an RE[...] object" 
AlgorithmCAD::usage="AlgorithmCAD[seq] uses " <>
  "Algorithm P from [Nuspl, Pillwein 2022] and decomposition into " <>
  "non-degenerate sequences to check whether the C-finite sequence is " <>
  "positive. Returns True if the sequence " <>
  "could be proven to be positive (i.e., every term of the sequence is " <>
  "positive) and False if a negative term was found \n\n" <>
  "seq... a C-finite sequence as an RE[...] object \n\n" <>
  "Options: \n" <>
  "Strict -> if True, positivity is checked, if False, non-negativity " <>
  " is checked (default: True). \n" <>
  "Verbose -> displays information during the execution of the algorithm " <>
  "(default: False)"
AlgorithmClassic::usage="AlgorithmClassic[seq] uses " <>
  "Algorithm C from [Nuspl, Pillwein 2022] and decomposition into " <>
  "non-degenerate sequences to check whether the C-finite sequence is " <>
  "positive.  Returns True if the sequence " <>
  "could be proven to be positive (i.e., every term of the sequence is " <>
  "positive) and False if a negative term was found \n\n" <>
  "seq... a C-finite sequence as an RE[...] object \n\n" <>
  "Options: \n" <>
  "Strict -> if True, positivity is checked, if False, non-negativity " <>
  " is checked (default: True). \n" <>
  "Verbose -> displays information during the execution of the algorithm " <>
  "(default: False)"
AlgorithmDominantRootCAD::usage="AlgorithmDominantRootCAD[seq] " <>
  "uses Algorithm P from [Nuspl, Pillwein 2022] to check whether the " <>
  "C-finite sequence is positive. Returns True if the sequence " <>
  "could be proven to be positive (i.e., every term of the sequence is " <>
  "positive) and False if a negative term was found. " <>
  "If the sequence does not have a unique dominant root, an error is " <>
  "raised. \n\n" <>
  "seq... a C-finite sequence as an RE[...] object \n\n" <>
  "Options: \n" <>
  "Strict -> if True, positivity is checked, if False, non-negativity " <>
  " is checked (default: True). \n" <>
  "Verbose -> displays information during the execution of the algorithm " <>
  "(default: False)"
AlgorithmDominantRootClassic::usage="AlgorithmDominantRootClassic[seq] " <>
  "uses Algorithm C from [Nuspl, Pillwein 2022] to " <>
  "check whether the C-finite sequence is positive. " <>
  "Returns True if the sequence " <>
  "could be proven to be positive (i.e., every term of the sequence is " <>
  "positive) and False if a negative term was found." <>
  "If the sequence does not have a unique dominant root, an error is " <>
  "raised. \n\n" <>
  "seq... a C-finite sequence as an RE[...] object \n\n" <>
  "Options: \n" <>
  "Strict -> if True, positivity is checked, if False, non-negativity " <>
  " is checked (default: True). \n" <>
  "Verbose -> displays information during the execution of the algorithm " <>
  "(default: False)"
PositiveSequence::usage="PositiveSequence[seq] uses a combination of " <>
  "different algorithms to show " <>
  "positivity. If the sequence is D-finite, then Algorithms 1 and 2 " <>
  "from [Kauers, Pillwein 2010] are used. If the sequence is C-finite, " <>
  "it is checked whether Algorithms 1 and 2 from [Kauers, Pillwein 2010] " <>
  "yield a result in a short amount of time. If they do not, then the " <>
  "sequence is decomposed into non-degenerate sequences and Algorithm C " <>
  "from [Nuspl, Pillwein 2022] is used for each of these individual " <>
  "sequences. Returns True if the sequence " <>
  "could be proven to be positive (i.e., every term of the sequence is " <>
  "positive) and False if a negative term was found. \n\n" <>
  "seq... a D-finite sequence as an RE[...] object \n\n" <>
  "Options: \n" <>
  "Strict -> if True, positivity is checked, if False, non-negativity " <>
  " is checked (default: True). \n" <>
  "Verbose -> displays information during the execution of the algorithm " <>
  "(default: False)"
ZeroSequence::usage="ZeroSequence[] returns a zero sequence as an RE[...] " <>
 "object."

Strict::usage="Option for methods of PositiveSequence package."
Verbose::usage="Option for methods of PositiveSequence package."
Eventual::usage="Option for methods of PositiveSequence package."

(* Private Functions *)
(*
GetNonZeroTrailingCoefficient::usage=""
GetRootsMult::usage=""
GetRecurrence::usage=""
CutInitialValues::usage=""
GetCompanionMatrix::usage=""
GetShift::usage=""
GetExpSequence::usage=""
GetPhiAlgo1::usage=""
ConjugatePolynomial::usage=""
GetCharpoly::usage=""
PolyPositive::usage=""
GetDomRoots::usage=""
DecomposeApplyAlgo::usage=""
GetIntegerRoots::usage=""
IsCFinite::usage=""
*)

Begin["`Private`"]

(* ::Section:: *)
(* Private Helper Functions *)

(* Given a polynomial poly in variable var, returns a list of pairs *)
(* {zero, mult} of all the roots of the polynomial together with the *)
(* multiplicity.*)
(*GetRootsMult[poly_, var_, mult_ : True] := Module[
  {roots, rootsMult},

  If[poly == 0, Throw["Zero polynomial has infinitely many roots"]];
  If[Exponent[poly, var] == 0, Return[{}]];
  If[Exponent[poly, var] == 1,
    If[mult,
      Return[{{AlgebraicNumber[Last[Roots[poly==0, var]],{0,1}], 1}}],
      Return[{AlgebraicNumber[Last[Roots[poly==0, var]],{0,1}]}]
    ];
  ];
  roots = List@@Roots[poly==0, var(*, Cubics -> False, Quartics -> False*)];
  roots = ToNumberField[Table[Last[root], {root, roots}]];
  If[mult, Return[Tally[roots]], Return[DeleteDuplicates[roots]]];
]*)
GetRootsMult[poly_, var_, mult_ : True] := Module[
  {roots, rootsMult},

  If[poly == 0, 
    Throw["Zero polynomial has infinitely many roots", ZeroPolynomial]
  ];
  If[Exponent[poly, var] == 0, Return[{}]];
  If[Exponent[poly, var] == 1,
    If[mult,
      Return[{{Last[Roots[poly==0, var]], 1}}],
      Return[{Last[Roots[poly==0, var]]}]
    ];
  ];
  roots = List@@Roots[poly==0, var, Cubics -> False, Quartics -> False];
  roots = Table[Last[root], {root, roots}];
  If[mult, Return[Tally[roots]], Return[DeleteDuplicates[roots]]];
]

(* returns a list of integer roots *)
GetIntegerRoots[poly_, var_] := Module[
  {},

  Return[Select[GetRootsMult[poly, var, False], IntegerQ[RootReduce[#1]] &]];
]

(* Returns the list of coefficients of a sequence.*)
GetRecurrence[seq_] := Module[{},
  Return[seq[[1,1]][[2;;]]];
]

(* Returns the list of initial values of a sequence.*)
GetInitialValue[seq_] := Module[{},
  Return[seq[[1,2]]];
]

(* Returns the order of the sequence.*)
GetOrder[seq_] := Module[{},
  Return[Length[GetRecurrence[seq]]-1];
]


(* Returns a pair (N, seq2) such that seq2 has a non-zero *)
(* trailing coefficient and seq2[n]=seq[n+N] for all n.*)
GetNonZeroTrailingCoefficient[seq_] := Module[
  {coeffs, N, seq2, newCoeffs, order, initValues},

  coeffs = GetRecurrence[seq];
  For[N = 0, N < Length[coeffs], N += 1,
    If[coeffs[[N+1]]=!=0,Break[]];
  ];

  If[N==0,
    seq2 = seq,

    newCoeffs = coeffs[[N+1;;]];
    order = Length[newCoeffs] - 1;
    initValues = RE2L[seq, {N, N+order-1}];
    seq2 = RE[{Join[{0}, newCoeffs], initValues}, seq[[2]]];
  ];
  Return[{N, seq2}]
]

(* Given a sequence, returns the characteristic polynomial in the *)
(* given variable.*)
GetCharpoly[seq_, var_] := Module[
  {coeffsRec},

  coeffsRec = GetRecurrence[seq];
  Return[Sum[coeffsRec[[i+1]]*var^i, {i, 0, Length[coeffsRec]-1}]];
]

(* Returns a list of distinct dominant roots of the sequence  *)
GetDomRoots[seq_] := Module[
  {roots, domRoots, root, rootAbs, domRootAbs},

  If[seq == ZeroSequence[], Return[{0}]];

  roots = GetRootsMult[GetCharpoly[seq, x], x, False];
  domRoots = {roots[[1]]};
  domRootAbs = AlgebraicNumber[Abs[roots[[1]]], {0,1}];
  Do[

    rootAbs = AlgebraicNumber[Abs[root], {0,1}];
    If[rootAbs > domRootAbs,
      domRoots = {root};
      domRootAbs = rootAbs;
      Continue[];
    ];
    If[rootAbs == domRootAbs,
      AppendTo[domRoots, root];
    ],

    {root, roots[[2;;]]}
  ];
  Return[domRoots];
]

(* Returns the zero sequence *)
ZeroSequence[] := Module[{},
  Return[RE[{{0, 1}, {0}}, z[n]]]
]

(* Returns the sequence with only as many intial values specified as needed *)
CutInitialValues[seq_] := Module[{},
  Return[RE[{seq[[1,1]], seq[[1,2]][[;;GetOrder[seq]]]}, seq[[2]]]];
]

(* Returns the leading coefficient of the recurrence *)
GetLeadingCoefficient[seq_] := Module[{},
  Return[GetRecurrence[seq][[-1]]]
]

(* Returns the companion matrix of a sequence *)
GetCompanionMatrix[seq_] := Module[
  {M, order, lc, coeffs, i},

  lc = GetLeadingCoefficient[seq];
  coeffs = GetRecurrence[seq];
  order = GetOrder[seq];
  M = ConstantArray[0, {order, order}];

  For[i = 1, i <= order, i += 1,
    M[[i, order]] = -coeffs[[i]]/lc
  ];
  For[i = 2, i <= order, i += 1,
    M[[i, i-1]] = 1
  ];
  Return[M];
]

(* Returns a list of coefficients which are the coordinates of seq[n+i] *)
(* w.r.t. seq[n],...,seq[n+r-1] where r is the order of seq *)
GetShift[seq_, i_] := Module[
  {order, var, v},

  order = GetOrder[seq];
  If[i < order, Return[IdentityMatrix[order][[i+1]]]];

  var = (seq[[2]] /. x_[m_] -> m);
  v = (GetShift[seq, i-1] /. var -> var + 1);
  Return[GetCompanionMatrix[seq].v];
]

(* Creates a D-finite sequence poly(n)*base^n *)
GetExpSequence[poly_, base_, seqName_, var_] := Module[
  {shiftPoly, rec, initValues, numInitValues, i},

  If[poly == 0 || base == 0,
    Return[Return[RE[{{0, 1}, {0}}, seqName[var]]]]
  ];
  shiftPoly = (poly /. var -> (var+1));
  rec = {0, -shiftPoly*base, poly};
  numInitValues = Max[Append[GetIntegerRoots[poly, var], 0]];
  initValues = Table[(poly*base^i /. var -> i ), {i,0,numInitValues+1}];
  Return[RE[{rec, initValues}, seqName[var]]];
]

(* For a polynomial poly, returns an index n0 such that poly(n) >= (1-eps)^n *)
(* for all n >= n0. If this cannot be done, an error is raised *)
PolyPositive[poly_, var_, eps_ : 1] := Module[
  {lc, deriv, roots, n, i},

  lc = Coefficient[poly, var, Exponent[poly, var]];
  If[lc < 0, Throw["Polynomial is never positive", NegativePolynomial]];
  If[lc === 0, 
    If[eps === 1, 
      Return[0], 
      Throw["Polynomial is never positive", NegativePolynomial]
    ]
  ];

  deriv = D[poly, var];
  If[deriv === 0,
    If[eps === 1, Return[0]];
    Return[Max[0, Ceiling2[Log[1-eps, poly]]]];
  ];

  roots = Table[Ceiling[Last[root]], 
                  {root, Flatten[NSolve[deriv == 0, var, Reals]]}];
  n = Max[Append[roots, 0]];
  While[True,
    If[(poly /. var -> n) >= If[n === 0, 1, (1-eps)^n],
      For[i = n-1, i >= 0, i -= 1,
        If[(poly /. var -> i) < If[i === 0, 1, (1-eps)^i], Return[i+1]];
      ];
      Return[0];
    ];
    n += 1;
  ];
]

(* Given a sequence, create and evalute the expression Phi from Algo 1 from *)
(* [Kauers, Pillwein 2010] *)
CheckPhiAlgo1[seq_, rho_, strict_] := Module[
  {r, yVars, expr, cond1, cond2, x},

  r = GetOrder[seq];
  yVars = Table[y[i], {i, 0, r-1}];
  x = seq[[2,1]];
  If[strict,
    cond1 = And @@ Thread[yVars > 0] && x >= 0;
    cond2 = And @@ Table[GetShift[seq, i].yVars > 0, {i, r, rho-1}];
    expr = ForAll[Evaluate[Append[yVars, x]], 
                  Implies[cond1 && cond2, GetShift[seq, rho].yVars > 0]],

    cond1 = And @@ Thread[yVars >= 0] && x >= 0;
    cond2 = And @@ Table[GetShift[seq, i].yVars >= 0, {i, r, rho-1}];
    expr = ForAll[Evaluate[Append[yVars, x]], 
                  Implies[cond1 && cond2, GetShift[seq, rho].yVars >= 0]]
  ];
  Return[Reduce[expr, {}, Reals]];
]

(* Given a sequence, create and evalute the expression Phi from Algo 2 from *)
(* [Kauers, Pillwein 2010] *)
CheckPhiAlgo2[seq_, mu_, xi_, strict_] := Module[
  {r, yVars, expr, cond, cond1, cond2},

  r = GetOrder[seq];
  yVars = Table[y[i], {i, 0, r-1}];
  If[strict,
    cond = yVars[[1]] > 0,

    cond = yVars[[1]] >= 0
  ];
  cond1 = cond && (And @@ Table[yVars[[i]] >= mu*yVars[[i-1]],{i,2,r}]);
  cond2 = (GetShift[seq, r] /. {seq[[2,1]] -> x}).yVars >= mu*yVars[[r]];
  expr = ForAll[Evaluate[Append[yVars, x]], 
                Implies[x >= xi, Implies[cond1, cond2]]];

  Return[Reduce[expr, {xi, mu}, Reals]];
]

(* Given a closed form, returns {{poly, maxEv}, complexPart, realPart} where *)
(* complexPart and realPart are lists with entries (poly, ev) *)
(* throws an exception if there is no unique dominant eigenvalue *)
SplitClosedForm[closedForm_] := Module[
  {term, maxEV, maxEVAbs, maxPoly, termEVAbs, numDomEV = 1, ev, 
   poly, complexPart = {}, realPart = {}, complexEVS = {}},

  maxEV = closedForm[[1,2]];
  maxEVAbs = AlgebraicNumber[Abs[maxEV], {0,1}];
  maxPoly = closedForm[[1,1]];

  (* Determine number of dominant eigenvalues *)
  Do[
    termEVAbs = AlgebraicNumber[Abs[term[[2]]], {0,1}];
    If[termEVAbs > maxEVAbs,
      numDomEV = 1;
      maxPoly = term[[1]];
      maxEV = term[[2]];
      maxEVAbs = termEVAbs,
      If[termEVAbs == maxEVAbs,
        numDomEV += 1
      ]
    ],

    {term, closedForm[[2;;]]}];

  If[numDomEV > 1, 
    Throw["More than one dominant eigenvalue", NonUniqueDominantEigenvalue]
  ];

  Do[

    {poly, ev} = term;
    If[ev == maxEV, Continue[]];
    If[Element[ev, Reals],

      AppendTo[realPart, {poly, ev}],

      (* eigenvalue is complex *)
      If[Not[MemberQ[complexEVS, ev]],
        AppendTo[complexEVS, ev];
        AppendTo[complexEVS, Conjugate[ev]];
        AppendTo[complexPart, {poly, ev}]
      ];
    ],

    {term, closedForm}];

  Return[{{maxPoly, maxEV}, complexPart, realPart}];
]

(* Computes the complex conjugate of the polynomial poly(x) *)
ConjugatePolynomial[poly_, x_] := Module[
  {i, coeffs},

  coeffs = CoefficientList[poly, x];
  Return[Sum[Conjugate[coeffs[[i]]]*x^(i-1), {i, 1, Length[coeffs]}]];
]

(* Root reduces all the initial values and recurrence coefficients of the *)
(* sequence *)
RootReduceSeq[seq_] := Module[
  {},

  Return[RE[{RootReduce@seq[[1,1]],RootReduce@seq[[1,2]]},seq[[2]]]];
]

(* given a number x, computes an integer greater than x; this might not be *) 
(* the smallest one this can be used if Ceiling tries to numerically prove *)
(* that an expression is already an integre, e.g. Ceiling[Log[8]/Log[2]] *)
Ceiling2[x_] := Module[
  {},

  Quiet@Check[
    Return[Ceiling[x]],
    Return[Ceiling[x + 1/2]]
  ];
]

(* ::Section:: *)
(* Public helper functions *)

(* Returns a list of pairs {poly, ev} describing the closed form of  *)
(* the C-finite sequence as a sum of polynomials times exponential sequences *)
(* the polynomials are polynomials in the given variable n *)
CFiniteClosedForm[seq_, n_] := Module[
  {seqShifted, roots, var, numVars, M, k, row, rhs, sol, closedForm = {},
  j, i, coeffs, poly},

  If[seq == ZeroSequence[], Return[{{0, 0}}]];
  seqShifted = GetNonZeroTrailingCoefficient[seq][[2]];

  (* set up and solve linear system *)
  roots = GetRootsMult[GetCharpoly[seqShifted, var], var];
  numVars = Sum[root[[2]], {root, roots}];
  M = ConstantArray[0, {numVars, numVars}];
  For[k = 0, k < numVars, k += 1,
    row = Flatten[Table[If[k==0 && j == 0, 1, k^j]*root[[1]]^k , 
                        {root, roots}, {j, 0, root[[2]]-1}]];
    M[[k+1]] = row;
  ];
  rhs = RE2L[seqShifted, numVars-1];
  sol = LinearSolve[M, rhs];
  (* assemble list of closed form *)
  i = 0;
  Do[
    coeffs = sol[[i+1;;(i+root[[2]])]];
    poly = Sum[RootReduce[coeffs[[j+1]]]*n^j,{j,0,Length[coeffs]-1}];
    AppendTo[closedForm, {poly, RootReduce[root[[1]]]}];
    i += root[[2]],

    {root, roots}];

  Return[Select[closedForm, (#1[[2]] =!= 0 && #1[[1]] =!= 0) & ]];
]

(* given a symbolic expression in one variable, creates 100 terms, *)
(* guesses a C-finite recurrence from these terms and returns the sequence *)
SequenceFromExpression[expr_, seqName_, var_] := Module[
  {vars, data, numTerms=100, n, seq, initValues, maxOrder=50},

  vars = DeleteDuplicates@Cases[expr, _Symbol, Infinity];
  If[Length[vars] > 1, 
    Throw["Expression contains more than one variable", NotUnivariate]
  ];
  If[Length[vars] == 1,
    data = Table[expr/.vars[[1]] -> n, {n, 0, numTerms-1}],
    data = Table[expr, {n, 0, numTerms-1}]
  ];
  seq = GuessRE[data, seqName[var], {0,maxOrder},{0,0}, Transform->{"ogf"}][[1]];
  Return[CutInitialValues[DefineS[seq, seqName[var]]]];
]

(* given a symbolic expression in one variable, creates 100 terms, *)
(* guesses a C-finite recurrence from these terms and returns the sequence *)
SequenceFromExpression[expr_, name_] := Module[
  {fName, arg, split},

  split = (name /. fName_[arg_] -> {fName, arg});
  Return[SequenceFromExpression[expr, split[[1]], split[[2]]]];
]

(* Checks whether the C-finite sequence is degenerate, i.e. if a quotient of *)
(* the eigenvalues is a root of unity *)
IsDegenerate[seq_] := Module[
  {roots, pair, rootOfUnityFound = False},

  If[seq == ZeroSequence[], Return[False]];
  roots = GetRootsMult[GetCharpoly[seq, x], x, False];
  Do[
    If[pair[[1]] == 0 || pair[[2]] == 0, Continue[]];
    If[RootOfUnityQ[RootReduce[pair[[1]]/pair[[2]]]], 
      rootOfUnityFound=True;Break[]
    ],

    {pair, Subsets[roots, {2}]}
  ];

  Return[rootOfUnityFound];
]

(* Checks whether the sequence is C-finite, i.e., the recurrence only contains *)
(* constants *)
IsCFinite[seq_] := Module[
  {rec},

  rec = GetRecurrence[seq];
  Return[AllTrue[rec, NumberQ]];
]

(* Returns a list of non-degernate (or zero) sequences such that the sequence *)
(* is an interlacing of those sequences *)
DecomposeDegenerate[seq_] := Module[
  {k = 1, seqs, v, var},

  seqs = {seq};
  var = seq[[2, 1]];
  cond[seq2_] := Return[Not[IsDegenerate[seq2]] || ZeroSequence[] == seq2];
  While[Not[AllTrue[seqs, cond]],
    k += 1;
    seqs = Table[RESubsequence[seq, k*var+v], {v, 0, k-1}]
  ];
  Return[seqs];
]

(* ::Section:: *)

(* Implement Algorithm1e, a variation of Algorithm 1 from [KP10] *)
Options[KPAlgorithm1] = {Strict->True, Eventual->True, Verbose->False};
KPAlgorithm1[seq_, opts___] := Module[
  {n = 0, n0 = 0, d = seq, r = GetOrder[seq], val, cond},

  {strict, ev, verbose} = {Strict, Eventual, Verbose} /. {opts} 
                              /. Options[KPAlgorithm1];
  While[n < r || CheckPhiAlgo1[d, n, strict]===False,
    val = RE2L[d, d[[2]], {n}][[1]];
    If[If[strict, RootReduce[val] > 0, RootReduce[val] >= 0],

      If[verbose, Print["n=", n, ", increase induction hypothesis"];];
      n = n + 1,

      If[verbose, Print["Negative term found, change sequence"];];
      If[Not[ev], Return[False]];
      n = n + 1;
      n0 = n0 + n;
      d = Shift[d, n];
      n = 0;
    ];
  ];
  If[Not[ev],
    cond = If[strict, Positive[#1] & , NonNegative[#1] &];
    If[verbose, Print["Check ", n, " initial terms"];];
    If[AllTrue[RE2L[seq, seq[[2]], n], cond],
      Return[True],
      Return[False]
    ];
  ];
  Return[n0];
]

(* Implement Algorithm2e, a variation of Algorithm 2 from [KP10] *)
Options[KPAlgorithm2] = {Strict->True, Eventual->True, Verbose->False};
KPAlgorithm2[seq_,  opts___] := Module[
  {n = 0, n0 = 0, d = seq, r = GetOrder[seq], val, cond1, cond2, 
   fullCond, i, Psi, cond},

  {strict, ev, verbose} = {Strict, Eventual, Verbose} /. {opts} 
                                /. Options[KPAlgorithm2];
  Psi = CheckPhiAlgo2[d, mu, xi, strict];
  While[True,
    val = RE2L[d, d[[2]], {n}][[1]];
    If[If[strict, RootReduce[val] > 0, RootReduce[val] >= 0],
      cond1 = And@@Table[
                RE2L[d, d[[2]], {i+1}][[1]] >= mu*RE2L[d,d[[2]],{i}][[1]],
              {i, n, n+r-2}];
      cond2 = (Psi /. xi -> n) && cond1;
      If[verbose, Print["n=", n, ", check condition"];];
      fullCond = Reduce[Exists[mu, If[strict, mu > 0, mu >= 0] && cond2], 
                        {}, Reals];
      If[fullCond === True,
        If[Not[ev],
          cond = If[strict, Positive[#1] & , NonNegative[#1] &];
          If[verbose, Print["Check ", n, " initial terms"];];
          If[AllTrue[RE2L[seq, seq[[2]], n], cond],
            Return[True],
            Return[False]
          ];
        ];
        Return[n0],

        n = n+1;
      ],

      If[Not[ev], Return[False]];
      If[verbose, Print["Negative term found, change sequence"];];
      n = n + 1;
      n0 = n0 + n;
      d = Shift[d, n];
      Psi = CheckPhiAlgo2[d, mu, xi, strict];
      n = 0;
    ];
  ];
]

(* decomposes the sequence into non-degenerate sequences and applies *)
(* the given function to each part *)
Options[DecomposeApplyAlgo] = {Strict->True, Verbose->False};
DecomposeApplyAlgo[seq_, algo_, opts___] := Module[
  {seqs, c, ret = True},

  {strict, verbose} = {Strict, Verbose} /. {opts} /. Options[DecomposeApplyAlgo];
  seqs = DecomposeDegenerate[seq];
  Do[

    If[c == ZeroSequence[], If[strict, ret=False; Break[]]];
    If[Not[algo[c, Strict->strict, Verbose->verbose]], ret=False; Break[]],

    {c, seqs}];
  Return[ret];
]

(* decomposes a sequence into non-degenerate sequences and uses *)
(* AlgorithmDominantRootCAD for each of those *)
Options[AlgorithmCAD] = {Strict->True, Verbose->False};
AlgorithmCAD[seq_, opts___] := Module[
  {},

  {strict, verbose} = {Strict, Verbose} /. {opts} /. Options[AlgorithmCAD];
  Return[DecomposeApplyAlgo[seq, AlgorithmDominantRootCAD, 
                            Strict->strict, Verbose->verbose]];
]

(* decomposes a sequence into non-degenerate sequences and uses *)
(* AlgorithmDominantRootClassic for each of those *)
Options[AlgorithmClassic] = {Strict->True, Verbose->False};
AlgorithmClassic[seq_, opts___] := Module[
  {},

  {strict, verbose} = {Strict, Verbose} /. {opts} /. Options[AlgorithmClassic];
  Return[DecomposeApplyAlgo[seq, AlgorithmDominantRootClassic, 
                            Strict->strict, Verbose->verbose]];
]

(* Computes the closed form of a C-finite sequence and uses KPAlgorithm1 *)
(* and KPAlgorithm2 on the  dominant term and all other terms *)
Options[AlgorithmDominantRootCAD] = {Strict->True, Verbose->False};
AlgorithmDominantRootCAD[seq_, opts___] := Module[
  {closedForm, nTC, nCheck = 0, maxEv, posSeq, fac, seq2, seq3, 
   seq4, poly, ev, cond, seqCheck},

  {strict, verbose} = {Strict, Verbose} /. {opts} 
                            /. Options[AlgorithmDominantRootCAD];
  {nTC, seq2} = GetNonZeroTrailingCoefficient[seq];
  If[verbose, Print["Compute closed form"]];
  closedForm = CFiniteClosedForm[seq2, n];
  If[verbose, Print["Closed form computed"]];

  (* Check if we have unique dominating real positive root *)
  {{poly, maxEv}, complexPart, realPart} = SplitClosedForm[closedForm];
  If[verbose, Print["Closed form split into complex and real part"]];
  If[Not[Element[maxEv, Reals]],
    Throw["Dominant eigenvalue is not real", NonRealDominantEigenvalue],

    If[RootReduce[maxEv] < 0, Return[False]];
    If[RootReduce[maxEv] == 0, Return[Not[strict]]];
  ];
  fac = Length[complexPart] + Length[realPart];

  (* Special case for order 1 sequence *)
  If[fac == 0, Return[RootReduce[maxEv] > 0 && PolyPositive[poly, n] == 0]];

  (* Apply algorithms for eventual positivity *)
  seq3 = GetExpSequence[poly/fac, 1, f3, n];
  If[verbose, Print["Check positivity of the terms"]];
  Do[
    {poly, ev} = term;
    seq4 = GetExpSequence[poly, RootReduce[ev/maxEv], f4, n];
    seq4 += GetExpSequence[ConjugatePolynomial[poly, n], 
                           RootReduce[Conjugate[ev]/maxEv], f5, n];
    If[verbose, Print["Compute sum of sequences"]];
    seqCheck = RootReduceSeq[seq3+seq4];
    If[verbose, Print["Check positivity of complex term"]];
    (* Print["Check positivity of ", seq3+seq4]; *)
    nCheck = Max[nCheck, KPAlgorithm1[seqCheck, Strict->strict]],

    {term, complexPart}
  ];
  Do[
    {poly, ev} = term;
    seq4 = GetExpSequence[poly, RootReduce[ev/maxEv], f4, n];
    If[verbose, Print["Compute sum of sequences"]];
    seqCheck = RootReduceSeq[seq3+seq4];
    If[verbose, Print["Check positivity of real term"]];
    nCheck = Max[nCheck, KPAlgorithm2[seqCheck, Strict->strict]],

    {term, realPart}
  ];
  cond = If[strict, Positive[RootReduce[#1]] & , NonNegative[RootReduce[#1]] &];
  If[verbose, Print["Check ", nTC+nCheck+1, " initial terms"];];
  If[AllTrue[RE2L[seq, seq[[2]], nTC + nCheck], cond],
    Return[True],
    Return[False]
  ];
]

(* Uses a classical method for showing positivity of a sequence using *)
(* bounds derived from the closed form *)
Options[AlgorithmDominantRootClassic] = {Strict->True, Verbose->False};
AlgorithmDominantRootClassic[seq_, opts___] := Module[
  {n, c, domRoots, domRoot, cond},

  {strict, verbose} = {Strict, Verbose} /. {opts} 
                        /. Options[AlgorithmDominantRootClassic];
  BoundPoly[poly_, var_] := Module[
    {maxCoeff},

    If[poly == 0, Return[0]];
    maxCoeff = Max[RootReduce@Abs@CoefficientList[poly, var]];
    Return[Ceiling2[maxCoeff*(Exponent[poly, var] + 1)]];
  ];

  GetConstants[s_, maxRoot_, var_] := Module[
    {closedForm, p2, term, d2, c2, l2, rootAbs},

    closedForm = CFiniteClosedForm[s, var];
    p2 = Select[closedForm, RootReduce[#1[[2]]] == RootReduce[maxRoot] &][[1,1]];
    closedForm = Select[closedForm, #1[[2]] != maxRoot & ];
    If[Length[closedForm] == 0, Return[{1, 0, 1/2, p2}]];
    d2 = Max[Table[Exponent[term[[1]], var], {term, closedForm}]];
    c2 = Sum[BoundPoly[term[[1]], var], {term, closedForm}];
    l2 = RootReduce[Abs[closedForm[[1, 2]]]];
    Do[

      rootAbs = RootReduce[Abs[term[[2]]]];
      If[RootReduce[rootAbs - l2] > 0, l2 = rootAbs],

      {term, closedForm[[2;;]]}
    ];
    Return[{c2, d2, RootReduce[l2/maxRoot], p2}];
  ];

  GetBound[s_, maxRoot_, var_] := Module[
    {c2, d2, l2, p2, eps, f2, boundPoly, n2, lhs},

    {c2, d2, l2, p2} = GetConstants[s, maxRoot, var];
    (*Print[c2, " ", d2, " ", l2, " ", p2];*)
    c2 = Ceiling2[c2];

    eps = RootReduce[(1-l2)/2];
    f2 = If[d2 == 0, RootReduce[(l2+eps)/l2], RootReduce[((l2+eps)/l2)^(1/d2)]];

    boundPoly = PolyPositive[p2, var, eps];
    If[d2 == 0,
      Return[Max[Ceiling2[RootReduce[Log[c2]/Log[f2]]], boundPoly]],

      n2 = Max[1, Ceiling2[RootReduce[1/Log[f2]]]];
      lhs = Ceiling2[RootReduce[Log[c2^(1/d2)]/Log[f2]]];
      While[True,
        If[0 < RootReduce[n2-Log[n2]/Log[f2]-lhs],
          Return[Max[n2, boundPoly], Module];
        ];
        n2 += 1;
      ];
    ];
  ];

  cond = If[strict, Positive[RootReduce[#1]] & , NonNegative[RootReduce[#1]] &];
  If[verbose, Print["Check if trailing coefficient is zero"]];
  {n, c} = GetNonZeroTrailingCoefficient[seq];
  If[n > 0,
    If[verbose, Print["Use shifted sequence as trailing coefficient is zero"]];
    If[AllTrue[RE2L[seq, n + 1], cond],
      Return[AlgorithmDominantRootClassic[c, Strict->strict, Verbose->verbose]],
      Return[False]
    ];
  ];

  If[seq == ZeroSequence[], Return[Not[strict]]];
  domRoots = GetDomRoots[seq];
  If[verbose, Print["Dominating roots: ", domRoots]];
  If[Length[domRoots] > 1, 
    Throw["Sequence has not a unique dominant root",NonUniqueDominantEigenvalue];
    Return[]];
  domRoot = domRoots[[1]];
  If[domRoot <= 0, Return[False]];
  n = Max[GetBound[seq, domRoot, x], 1];
  If[verbose, Print["Check ", n+1, " initial terms"];];
  If[AllTrue[RE2L[seq, n + 1], cond],
    Return[True],
    Return[False]
  ]
]

Options[PositiveSequence] = {Strict->True, Verbose->False};
PositiveSequence[seq_, opts___] := Module[
  {time},

  {strict, verbose} = {Strict, Verbose} /. {opts} /. Options[PositiveSequence];
  If[Not[IsCFinite[seq]],
    (* sequence is assumed to be D-finite, not C-finite *)
    If[verbose, Print["Sequence is D-finite"];];
    time = 1;
    While[True,
      TimeConstrained[
        If[verbose, 
          Print["Try Algorithm 1 with time limit ", time, " seconds"];
        ];
        Return[KPAlgorithm1[seq, Strict->strict, 
                            Eventual->False, Verbose->verbose], Module]
      , time];
      TimeConstrained[
        If[verbose, 
          Print["Try Algorithm 2 with time limit ", time, " seconds"];
        ];
        Return[KPAlgorithm2[seq, Strict->strict, 
                            Eventual->False, Verbose->verbose], Module]
      , time];
      time = 5*time;
    ];
  ];

  (* sequence is C-finite *)
  TimeConstrained[
    If[verbose, Print["Try Algorithm 1"];];
    Return[KPAlgorithm1[seq, Strict->strict, 
                        Eventual->False, Verbose->verbose], Module]
  , 1];
  TimeConstrained[
    If[verbose, Print["Try Algorithm 2"];];
    Return[KPAlgorithm2[seq, Strict->strict, 
                        Eventual->False, Verbose->verbose], Module]
  , 1];
  If[verbose, Print["Try decomposition and classical algorithm"];];
  Return[AlgorithmClassic[seq, Strict->strict, Verbose->verbose], Module];

]

(* -------------------------------------------------------------------- *)

End[]


EndPackage[]
