import { marked } from 'marked'

type Item = { source?: string; title?: string; url?: string; published?: string }

export default function NewsCard({ md, items }: { md?: string, items?: Item[] }) {
  if (!md && (!items || !items.length)) return null
  return (
    <div className="card">
      <div className="section-title">Market News Summary</div>
      {md && <div style={{ marginBottom: 12 }} dangerouslySetInnerHTML={{ __html: marked.parse(md) as string }} />}
      {items && items.length > 0 && (
        <div>
          <div className="label" style={{ marginBottom: 8 }}>Headlines</div>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {items.map((it, idx) => (
              <li key={idx} style={{ marginBottom: 6 }}>
                <span className="badge" style={{ marginRight: 6 }}>{it.source}</span>
                <a href={it.url} target="_blank" rel="noreferrer">{it.title}</a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}