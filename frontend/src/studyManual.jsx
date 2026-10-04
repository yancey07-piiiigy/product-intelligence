import React,{useEffect,useState} from 'react'

/** Fixed manual-browsing condition for the image-first user study. */
export default function StudyManual({taskId}){
 const [page,setPage]=useState(1),[data,setData]=useState(null),[selected,setSelected]=useState(null),[choice,setChoice]=useState(null),[error,setError]=useState('')
 useEffect(()=>{let live=true;fetch(`/api/study/manual?task=${encodeURIComponent(taskId)}&page=${page}`).then(async r=>{const value=await r.json();if(!r.ok)throw Error(value.error||'Could not load task');return value}).then(v=>{if(live){setData(v);setError('')}}).catch(e=>{if(live)setError(e.message)});return()=>{live=false}},[taskId,page])
 if(error)return <main className="study-manual"><div className="error" role="alert">{error}</div></main>
 if(!data)return <main className="study-manual"><p>Loading the manual browsing task…</p></main>
 const task=data.task
 return <main className="study-manual"><header className="study-head"><div><span>SHOELENS USER STUDY · MANUAL CONDITION</span><h1>Find the exact shoe</h1><p>Compare the target photo with the catalog. You do not need to know its product name or ID. When ready, choose the item and report its catalog season and use.</p></div><div className="study-code">Task {task.task_id}</div></header>
 <section className="study-target"><img src={`/query/${task.query_id}`} alt="Target shoe to identify"/><div><h2>Target photo</h2><p>Browse products with the same catalog category, colour and audience. These clues are given to both study conditions.</p><dl><div><dt>Category</dt><dd>{task.articleType}</dd></div><div><dt>Colour</dt><dd>{task.baseColour}</dd></div><div><dt>Catalog audience</dt><dd>{task.gender}</dd></div></dl></div></section>
 {choice?<section className="study-choice"><h2>Choice recorded on this screen</h2><p>Selected record: <strong>#{choice.id}</strong> · {choice.productDisplayName}</p><p>Tell the facilitator this record ID, its catalog season and its catalog use. The facilitator records your time and answers. This page does not save your response.</p><button onClick={()=>setChoice(null)}>Change selection</button></section>:<>
 <div className="study-list-title"><h2>Catalog candidates</h2><span>{data.total} visually comparable catalog items · Page {data.page} of {Math.ceil(data.total/data.page_size)}</span></div>
 <div className="study-grid">{data.items.map(item=><button className="study-item" key={item.id} onClick={()=>setSelected(item)}><img src={`/image/${item.id}`} alt={item.productDisplayName}/><strong>{item.productDisplayName}</strong><small>Inspect record</small></button>)}</div>
 <div className="study-pagination"><button disabled={page===1} onClick={()=>setPage(p=>p-1)}>Previous</button><span>Page {page} / {Math.ceil(data.total/data.page_size)}</span><button disabled={page*data.page_size>=data.total} onClick={()=>setPage(p=>p+1)}>Next</button></div></>}
 {selected&&<div className="modal-backdrop" onClick={()=>setSelected(null)}><section className="modal panel study-modal" role="dialog" aria-modal="true" aria-label="Inspect catalog product" onClick={e=>e.stopPropagation()}><button className="close" onClick={()=>setSelected(null)}>Close ×</button><img src={`/image/${selected.id}`} alt={selected.productDisplayName}/><h2>{selected.productDisplayName}</h2><p>Catalog season: <strong>{selected.season||'Not recorded'}</strong></p><p>Catalog use: <strong>{selected.usage||'Not recorded'}</strong></p><button className="primary" onClick={()=>{setChoice(selected);setSelected(null)}}>Choose this product</button></section></div>}
 <p className="study-note">Manual condition: no search by product name or ID, no AI ranking. The candidate order is fixed for this task. The facilitator uses the same target-image reveal and time limit in both conditions.</p>
 </main>
}
