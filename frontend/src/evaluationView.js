export function comparisonRows(baseline,main,knownCount,totalCount){
 const percent=value=>value==null?'—':`${(value*100).toFixed(1)}%`
 return [
  {label:'Top-1 catalog match',description:'Correct first candidate among catalog shoes',baselineValue:baseline.known_top1,mainValue:main.known_top1,baselineCount:`${Math.round(baseline.known_top1*knownCount)} / ${knownCount}`,mainCount:`${Math.round(main.known_top1*knownCount)} / ${knownCount}`},
  {label:'Answers given',description:'Queries accepted as a specific product',baselineValue:baseline.coverage,mainValue:main.coverage,baselineCount:`${baseline.accepted_n??Math.round(baseline.coverage*totalCount)} / ${totalCount}`,mainCount:`${main.accepted_n??Math.round(main.coverage*totalCount)} / ${totalCount}`},
  {label:'Accuracy of answers given',description:'Correct identities among accepted answers',baselineValue:baseline.accepted_identity_accuracy,mainValue:main.accepted_identity_accuracy,baselineCount:baseline.accepted_n?'': 'No accepted matches',mainCount:`${main.correct_accepted_n??Math.round(main.accepted_identity_accuracy*main.accepted_n)} / ${main.accepted_n}`},
 ].map(row=>({...row,baselineText:percent(row.baselineValue),mainText:percent(row.mainValue)}))
}

export function decisionCounts(test){return {correct:test.correct_accepted_n,incorrect:test.accepted_n-test.correct_accepted_n,abstained:test.n-test.accepted_n}}
