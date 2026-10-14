import type { ApprovalRequest, ReadThirdPartySource, ThirdPartySource } from "../../../core/api";
import { asksAfterThirdParty } from "../../../shared/approvalRules";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";

/** The line for text its Answer read itself. */
const READ: Record<ReadThirdPartySource, MessageKey> = {
  imported: "approvals.thirdParty.read.imported",
  "web-page": "approvals.thirdParty.read.webPage",
  "checked-document": "approvals.thirdParty.read.checkedDocument",
};

/** The line for text in its Question context: in a Mind, or the Mind a Question in a conversation read. */
const IN_MIND: Record<ThirdPartySource, MessageKey> = {
  imported: "approvals.thirdParty.mind.imported",
  "web-page": "approvals.thirdParty.mind.webPage",
  "checked-document": "approvals.thirdParty.mind.checkedDocument",
};

/** The line for text in its Question context: in a conversation. */
const IN_CONVERSATION: Record<ThirdPartySource, MessageKey> = {
  imported: "approvals.thirdParty.conversation.imported",
  "web-page": "approvals.thirdParty.conversation.webPage",
  "checked-document": "approvals.thirdParty.conversation.checkedDocument",
};

/**
 * Why an approval card asks whatever the User's policy: its Answer had read
 * third-party text, and the call writes or sends data (#78). One line, for the
 * first reason: "This Answer read a web page, so it asks you first." Nothing
 * for a call that asks for another reason.
 */
export function ThirdPartyLine({ request }: { request: ApprovalRequest }) {
  const t = useT();
  const [reason] = request.thirdParty ?? [];
  if (!reason || !asksAfterThirdParty(request)) return null;
  const key =
    reason.via === "read"
      ? READ[reason.source]
      : (reason.via === "context" && request.mindId === null ? IN_CONVERSATION : IN_MIND)[
          reason.source
        ];
  return (
    <p
      data-testid="approval-third-party"
      data-via={reason.via}
      data-source={reason.source}
      className="approval-explanation"
    >
      {t(key)}
    </p>
  );
}
