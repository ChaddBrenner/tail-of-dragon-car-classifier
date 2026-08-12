import type { ImgHTMLAttributes } from "react";

type PhotoProps = Omit<ImgHTMLAttributes<HTMLImageElement>, "width" | "height"> & {
  eager?: boolean;
};

export function Photo({ className = "", eager = false, ...props }: PhotoProps) {
  return (
    <img
      {...props}
      className={`photo ${className}`.trim()}
      decoding="async"
      fetchPriority={eager ? "high" : "auto"}
      height="400"
      loading={eager ? "eager" : "lazy"}
      width="600"
    />
  );
}

export function PhotoSkeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden="true" className={`photo photoSkeleton ${className}`.trim()} />;
}
