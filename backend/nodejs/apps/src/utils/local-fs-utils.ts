import { ConnectorId } from '../libs/types/connector.types';

/**
 * Canonical Local FS connector key used across backend events/config.
 *
 * Local FS runs the same server-driven sync as every other connector: a
 * resync or scheduled tick reaches `run_sync`, which pulls file-event
 * metadata from the user's desktop through the desktop relay routes.
 */
export const LOCAL_FS_CONNECTOR_KEY = ConnectorId.LOCAL_FS as string;

/** Canonical Obsidian connector key (Python factory key `obsidian`). */
export const OBSIDIAN_CONNECTOR_KEY = ConnectorId.OBSIDIAN as string;

/**
 * Connector keys that sync a folder on the user's desktop through the desktop
 * relay: same owner-device claim, presence checks and 409 refusal codes.
 */
export const DESKTOP_FOLDER_CONNECTOR_KEYS: readonly string[] = Object.freeze([
  LOCAL_FS_CONNECTOR_KEY,
  OBSIDIAN_CONNECTOR_KEY,
]);

function normalizeConnectorName(connectorName: string): string {
  return connectorName
    .trim()
    .replace(/[_\s]+/g, '')
    .toLowerCase();
}

export function isLocalFsConnector(connectorName: string): boolean {
  return normalizeConnectorName(connectorName) === LOCAL_FS_CONNECTOR_KEY;
}

/**
 * The canonical key of a desktop-folder connector (`localfs`, `obsidian`), or
 * `null` when the name is not one. Accepts display names ("Local FS").
 */
export function desktopFolderConnectorKey(connectorName: string): string | null {
  const normalized = normalizeConnectorName(connectorName);
  return DESKTOP_FOLDER_CONNECTOR_KEYS.includes(normalized) ? normalized : null;
}

/** True for every connector served by the desktop relay (Local FS, Obsidian). */
export function isDesktopFolderConnector(connectorName: string): boolean {
  return desktopFolderConnectorKey(connectorName) !== null;
}
