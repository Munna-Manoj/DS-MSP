import { lazy, Suspense, type ReactNode } from "react";
import { href, useRoute, type Route } from "./router";
import { ExportPage } from "./studio/ExportPage";
import { JobPage } from "./studio/JobPage";
import { NewRunPage } from "./studio/NewRunPage";
import { RepeatPage } from "./studio/RepeatPage";
import { RunPage } from "./studio/RunPage";
import { RunsPage } from "./studio/RunsPage";
import { ErrorBox, Guard } from "./studio/ui";

const LearnPage = lazy(() => import("./learn/LearnPage").then((m) => ({ default: m.LearnPage })));

export function App() {
  const route = useRoute();
  const learn = route.page === "learn";
  return (
    <div className={`min-h-full flex flex-col ${learn ? "learn-root" : "studio-root"}`}>
      <Header route={route} />
      <div className="flex-1 flex flex-col">
       <Guard key={window.location.hash} fallback={(m) => <main className="mx-auto w-full max-w-3xl px-4 py-10"><ErrorBox title="This page could not be shown. Reload it, or go back to All results. Your calibrations are not affected." detail={m} /></main>}>
        {route.page === "learn" && (
          <Suspense fallback={<div className="p-8 text-muted">Loading…</div>}>
            <LearnPage />
          </Suspense>
        )}
        {route.page === "runs" && (__HAS_SERVER__ ? <RunsPage /> : <NoServer />)}
        {route.page === "run" && (__HAS_SERVER__ ? <RunPage id={route.id} cam={route.cam} /> : <NoServer />)}
        {route.page === "new" && (__HAS_SERVER__ ? <NewRunPage from={route.from} /> : <NoServer />)}
        {route.page === "job" && (__HAS_SERVER__ ? <JobPage id={route.id} /> : <NoServer />)}
        {route.page === "repeat" && (__HAS_SERVER__ ? <RepeatPage id={route.id} /> : <NoServer />)}
        {route.page === "export" && (__HAS_SERVER__ ? <ExportPage key={route.from ?? ""} from={route.from} /> : <NoServer />)}
       </Guard>
      </div>
    </div>
  );
}

function Header({ route }: { route: Route }) {
  const learn = route.page === "learn";
  const tab = (target: Route, label: string, active: boolean): ReactNode => (
    <a
      href={href(target)}
      aria-current={active ? "page" : undefined}
      className={`px-3 py-1.5 rounded-md text-sm font-medium whitespace-nowrap transition ${
        active
          ? learn ? "bg-panel-2 text-text" : "bg-s-accent-soft text-s-accent"
          : learn ? "text-muted hover:text-text" : "text-s-muted hover:text-s-text"
      }`}
    >
      {label}
    </a>
  );
  return (
    <header className={`border-b ${learn ? "border-line" : "border-s-rule bg-s-surface"}`}>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 h-14 flex items-center gap-4">
        <a href={href({ page: "runs" })} className="flex items-baseline gap-2 shrink-0">
          <span className={`font-bold tracking-tight ${learn ? "font-display text-lg" : "text-[17px]"}`}>
            DS-MSP
          </span>
          <span className={`text-sm hidden sm:inline ${learn ? "text-muted" : "text-s-muted"}`}>Studio</span>
        </a>
        <nav className="flex items-center gap-1 overflow-x-auto min-w-0" aria-label="Main">
          {tab({ page: "runs" }, "Results", route.page === "runs" || route.page === "run" || route.page === "repeat")}
          {__HAS_SERVER__ && tab({ page: "new" }, "New calibration", route.page === "new" || route.page === "job")}
          {__HAS_SERVER__ && tab({ page: "export" }, "Export", route.page === "export")}
          {tab({ page: "learn" }, "Learn", learn)}
        </nav>
        <a
          href="https://github.com/Munna-Manoj/DS-MSP"
          target="_blank"
          rel="noreferrer"
          className={`ml-auto text-sm hidden sm:inline ${learn ? "text-muted hover:text-text" : "text-s-muted hover:text-s-text"}`}
        >
          GitHub ↗
        </a>
      </div>
    </header>
  );
}

function NoServer() {
  return (
    <main className="mx-auto w-full max-w-3xl px-4 sm:px-6 py-12 grid gap-4">
      <h1 className="text-2xl font-semibold">Results open in your own Studio</h1>
      <p className="text-s-muted">
        This public page has no calibration results. Install DS-MSP with the interface and point
        it at a folder of results:
      </p>
      <pre className="font-studio-mono text-sm bg-s-sunken border border-s-rule rounded-lg p-4 overflow-x-auto">
        {`pip install "ds-msp[ui]"\nds-msp studio --root path/to/results`}
      </pre>
      <p className="text-s-muted">
        The <a className="text-s-accent underline" href={href({ page: "learn" })}>Learn</a> page
        works here: it shows how each camera model maps a 3D point to a pixel.
      </p>
    </main>
  );
}
