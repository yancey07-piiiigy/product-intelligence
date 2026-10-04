import test from 'node:test'
import assert from 'node:assert/strict'
import {comparisonRows,decisionCounts} from './evaluationView.js'

test('comparison keeps undefined baseline accuracy distinct from zero',()=>{
 const rows=comparisonRows({known_top1:15/42,coverage:0,accepted_identity_accuracy:null},{known_top1:41/42,coverage:42/60,accepted_identity_accuracy:41/42},42,60)
 assert.equal(rows[0].baselineCount,'15 / 42')
 assert.equal(rows[1].baselineCount,'0 / 60')
 assert.equal(rows[2].baselineCount,'No accepted matches')
 assert.equal(rows[2].baselineValue,null)
})

test('decision counts distinguish accepted errors and abstentions',()=>{
 assert.deepEqual(decisionCounts({n:60,accepted_n:42,correct_accepted_n:41}),{correct:41,incorrect:1,abstained:18})
})
