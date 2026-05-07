import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * シンプルなネイティブ <select> ベースのコンポーネント。
 * MVP 段階ではデザインの一貫性より機能優先で実装。
 * 将来 @radix-ui/react-select に差し替え可能。
 */
const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, children, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      "flex h-9 w-full rounded-md border border-[var(--color-input)] bg-[var(--color-background)] px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--color-ring)] disabled:cursor-not-allowed disabled:opacity-50",
      className,
    )}
    {...props}
  >
    {children}
  </select>
));
Select.displayName = "Select";

export { Select };
