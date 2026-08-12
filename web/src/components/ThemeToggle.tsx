import { useEffect, useState } from "react";
import type { Theme } from "../types";
import { MoonIcon, SunIcon } from "./Icons";

const THEME_KEY = "tail-dragon-theme-v1";

function readTheme(): Theme {
  const current = document.documentElement.dataset.theme;
  return current === "light" ? "light" : "dark";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const nextTheme = theme === "dark" ? "light" : "dark";

  return (
    <button
      aria-label={`Switch to ${nextTheme} theme`}
      aria-pressed={theme === "light"}
      className="themeToggle"
      data-testid="theme-toggle"
      onClick={() => setTheme(nextTheme)}
      type="button"
    >
      <SunIcon />
      <span aria-hidden="true" className={`themeKnob ${theme}`} />
      <MoonIcon />
    </button>
  );
}
