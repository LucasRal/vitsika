"use client";
import { useEffect, useMemo, useState } from "react";
import { CircleMarker, MapContainer, Marker, TileLayer, Tooltip, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";

const MADAGASCAR: L.LatLngBoundsExpression = [[-25.7, 43.0], [-11.8, 50.7]];
const CLUSTER_ABOVE = 200;   // individual markers up to here, grid clusters beyond
const CELL_PX = 56;

type Props = { points: [number, number][]; colour: string; genus: string };
type Cluster = { lat: number; lon: number; n: number };

/** Grid clustering in screen space at the current zoom — no plugin needed. */
function clusterPoints(map: L.Map, points: [number, number][]): Cluster[] {
  const cells = new Map<string, { lat: number; lon: number; n: number }>();
  for (const [lat, lon] of points) {
    const p = map.project([lat, lon], map.getZoom());
    const key = `${Math.floor(p.x / CELL_PX)}:${Math.floor(p.y / CELL_PX)}`;
    const c = cells.get(key);
    if (c) { c.lat += lat; c.lon += lon; c.n += 1; } else cells.set(key, { lat, lon, n: 1 });
  }
  return Array.from(cells.values()).map((c) => ({ lat: c.lat / c.n, lon: c.lon / c.n, n: c.n }));
}

function Clusters({ points, colour }: { points: [number, number][]; colour: string }) {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());
  useMapEvents({ zoomend: () => setZoom(map.getZoom()) });
  const clusters = useMemo(() => clusterPoints(map, points), [map, points, zoom]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      {clusters.map((c, i) => c.n === 1 ? (
        <CircleMarker key={i} center={[c.lat, c.lon]} radius={4} pathOptions={{ color: "#fff", weight: 1, fillColor: colour, fillOpacity: 0.85 }} />
      ) : (
        <Marker key={i} position={[c.lat, c.lon]} eventHandlers={{ click: () => map.setView([c.lat, c.lon], Math.min(map.getZoom() + 2, 12)) }}
                icon={L.divIcon({ className: "leaflet-div-icon", html: `<div class="cluster" style="background:${colour}">${c.n}</div>`,
                                  iconSize: [Math.min(44, 22 + Math.log2(c.n) * 4), Math.min(44, 22 + Math.log2(c.n) * 4)] })}>
          <Tooltip>{c.n} specimens — click to zoom</Tooltip>
        </Marker>
      ))}
    </>
  );
}

function FitBounds({ points }: { points: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length === 0) { map.fitBounds(MADAGASCAR); return; }
    map.fitBounds(L.latLngBounds(points).pad(0.15), { maxZoom: 9 });
  }, [map, points]);
  return null;
}

export function GeoMap({ points, colour, genus }: Props) {
  const cluster = points.length > CLUSTER_ABOVE;
  return (
    <div className="hairline relative overflow-hidden rounded-md">
      <MapContainer bounds={MADAGASCAR} style={{ height: 520 }} scrollWheelZoom={false} attributionControl>
        <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                   url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <FitBounds points={points} />
        {cluster ? <Clusters points={points} colour={colour} /> : points.map(([lat, lon], i) => (
          <CircleMarker key={i} center={[lat, lon]} radius={4.5} pathOptions={{ color: "#fff", weight: 1, fillColor: colour, fillOpacity: 0.85 }}>
            <Tooltip><span className="genus">{genus}</span> · {lat.toFixed(3)}, {lon.toFixed(3)}</Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
      <div className="pointer-events-none absolute right-2 top-2 z-[400] rounded-sm bg-surface/90 px-2 py-1 text-xs text-ink-2">
        <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: colour }} />
        <span className="genus">{genus}</span> · {points.length.toLocaleString()} georeferenced specimens{cluster ? " · clustered, click to zoom" : ""}
      </div>
    </div>
  );
}
