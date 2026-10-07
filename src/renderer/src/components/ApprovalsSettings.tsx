import { useEffect, useState } from "react";
import type { ApprovalPolicy } from "../../../core/api";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";

/**
 * Settings → Approvals: every policy that replaces a default, so the User
 * can tighten permissions later. Each Tool always allowed, each Tool set to
 * ask every time (even one its Connector says only reads), and (#41) each
 * Skill whose scripts always run; each can be revoked, back to the default.
 */
export function ApprovalsSettings() {
  const t = useT();
  const [policies, setPolicies] = useState<ApprovalPolicy[] | null>(null);

  useEffect(() => {
    core.listApprovalPolicies().then(setPolicies, () => undefined);
    return core.on("approvals.changed", setPolicies);
  }, []);

  if (!policies) return null;
  const revoke = (policy: ApprovalPolicy) =>
    core.revokeApprovalPolicy(policy.id).catch((failure: unknown) => {
      useAppStore.setState({ actionError: errorMessage(failure) });
    });

  return (
    <section data-testid="approvals-settings">
      <h3 className="mb-1 text-sm font-medium">{t("approvals.settings.title")}</h3>
      <p className="text-sm text-gray-600">{t("approvals.settings.body")}</p>
      {policies.length === 0 ? (
        <p className="mt-2 text-sm text-gray-500">{t("approvals.settings.empty")}</p>
      ) : (
        <ul className="mt-2 flex flex-col gap-1">
          {policies.map((policy) => {
            const { subject } = policy;
            const name =
              subject.kind === "tool"
                ? `${policy.ownerName} · ${subject.tool}`
                : t("approvals.settings.skillScripts", { skill: policy.ownerName });
            const value =
              subject.kind === "skill-script"
                ? t("approvals.policy.alwaysRun")
                : policy.policy === "always"
                  ? t("approvals.policy.always")
                  : t("approvals.policy.ask");
            return (
              <li
                key={policy.id}
                data-testid="approval-policy"
                data-policy={policy.policy}
                className="flex items-center gap-2 rounded-[9px] border border-gray-200 px-3 py-1.5 text-sm"
              >
                <span className="min-w-0 flex-1 truncate" title={name}>
                  {subject.kind === "tool" ? (
                    <>
                      {policy.ownerName} · <span className="font-mono">{subject.tool}</span>
                    </>
                  ) : (
                    name
                  )}
                </span>
                <span
                  data-testid="approval-policy-value"
                  className={`shrink-0 text-xs ${policy.policy === "always" ? "text-amber-700" : "text-gray-500"}`}
                >
                  {value}
                </span>
                <button
                  type="button"
                  data-testid="approval-policy-revoke"
                  aria-label={t("approvals.settings.revokeLabel", { name })}
                  onClick={() => void revoke(policy)}
                  className="shrink-0 rounded-[6px] border border-gray-300 px-2 py-0.5 text-xs hover:bg-gray-100"
                >
                  {t("approvals.settings.revoke")}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
