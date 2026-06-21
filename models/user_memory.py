"""Couche service pour les memoires utilisateur stockees dans Supabase.

Ce module expose :class:`UserMemoryService`, une fine couche de logique
metier au-dessus du SDK Python de Supabase. Elle centralise tous les acces
CRUD a la table ``user_memories`` et garantit que chaque operation de
lecture, mise a jour ou suppression est bien limitee a un seul
``user_id``.

Pourquoi filtrer deux fois par ``user_id`` meme avec le client admin ?
------------------------------------------------------------------------
Le client Supabase utilise ici est le client **admin** (voir
``get_supabase_admin_client``), qui s'authentifie avec la cle de service
("service role") et contourne donc entierement les politiques Row Level
Security (RLS). Cela signifie que la base de donnees elle-meme
n'empechera PAS un utilisateur de lire ou de supprimer les lignes d'un
autre utilisateur si l'on oublie de filtrer manuellement.

Pour compenser, chaque requete de ce service enchaine explicitement
``.eq("user_id", ...)`` en plus de ``.eq("id", ...)``. Il s'agit d'un
pattern de defense en profondeur : meme si les RLS sont mal configurees,
desactivees, ou jamais appliquees a cette table, la couche applicative
continue d'imposer l'isolation entre utilisateurs (multi-tenant). Si vous
ajoutez une nouvelle methode a cette classe, conservez toujours ce double
filtrage.

Fonctions utilitaires au niveau du module
-------------------------------------------
En bas du fichier, un petit ensemble de fonctions libres
(``create_memory``, ``get_user_memories``, etc.) encapsulent une instance
par defaut de :class:`UserMemoryService`, creee de maniere paresseuse
(lazy). Elles existent pour que les appelants (par exemple les handlers
de routes FastAPI) puissent faire
``from services.user_memories import get_memory`` sans avoir a construire
manuellement une instance du service a chaque fois. L'instance est creee
au premier usage plutot qu'a l'import, ce qui evite tout effet de bord a
l'import du module (utile pour les tests unitaires qui simulent le client
Supabase avant le premier appel).
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from core.supabase_client import get_supabase_admin_client
from models.user_memory import UserMemory, UserMemoryCreate, UserMemorySource, UserMemoryUpdate

logger = logging.getLogger("makenbrain.user_memories")

USER_MEMORIES_TABLE = "user_memories"


class UserMemoryServiceError(RuntimeError):
    """Levee quand une operation sur une memoire utilisateur echoue, au
    niveau du service ou de Supabase.

    Il s'agit de l'exception de base du module. Elle encapsule les
    erreurs de plus bas niveau (erreurs reseau, reponses Supabase mal
    formees, exceptions du SDK, ...) afin que les appelants n'aient
    besoin d'intercepter qu'un seul type d'exception au lieu de deviner
    ce que le SDK Supabase pourrait lever.
    """


class UserMemoryNotFoundError(UserMemoryServiceError):
    """Levee quand la memoire demandee n'existe pas dans le perimetre de
    l'utilisateur.

    Notez que cette exception est aussi levee quand la memoire existe
    dans la table mais appartient a un *autre* utilisateur. Du point de
    vue de l'appelant, ce cas est volontairement indiscernable de
    "n'existe pas" : on ne veut jamais reveler l'existence des donnees
    d'un autre utilisateur via les messages d'erreur.
    """


class UserMemoryService:
    """Service metier pour les memoires appartenant a un profil
    utilisateur Supabase.

    Chaque methode publique prend un ``user_id`` en premier argument et
    l'utilise pour delimiter la requete Supabase sous-jacente. Une seule
    instance du service peut donc etre reutilisee en toute securite entre
    plusieurs requetes pour des utilisateurs differents (l'instance
    elle-meme ne conserve aucun etat propre a un utilisateur, uniquement
    la connexion au client Supabase).
    """

    def __init__(self, client: Any | None = None) -> None:
        """Cree le service avec, en option, un client Supabase personnalise.

        Args:
            client: Un client Supabase deja configure, a utiliser a la
                place du client admin par defaut. Principalement utile
                pour les tests unitaires, ou un client factice/mock peut
                etre injecte pour eviter d'appeler un veritable projet
                Supabase.
        """
        self._client = client or get_supabase_admin_client()

    def create_memory(
        self,
        user_id: UUID | str,
        content: str,
        source: UserMemorySource | str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Cree une memoire pour un utilisateur et retourne un dict normalise.

        Args:
            user_id: Proprietaire de la nouvelle memoire. Stocke sur la
                ligne afin que toutes les lectures/mises a jour/
                suppressions futures puissent etre limitees a cet
                utilisateur.
            content: Le texte de la memoire elle-meme (par exemple un
                fait ou une note a retenir sur l'utilisateur).
            source: D'ou provient cette memoire (voir
                :class:`UserMemorySource` — par exemple ajoutee
                manuellement par l'utilisateur vs. deduite
                automatiquement d'une conversation).
            metadata: Metadonnees JSON libres optionnelles attachees a la
                memoire (tags, identifiant de conversation, score de
                confiance, etc.). Par defaut, un dict vide si non fourni.

        Returns:
            Un dict serialisable en JSON representant la ligne nouvellement
            creee, valide et normalise via le modele :class:`UserMemory`
            (c'est exactement ce qui doit etre retourne directement par
            un endpoint API).

        Raises:
            UserMemoryServiceError: Si l'appel d'insertion Supabase leve
                une exception, ou s'il reussit mais ne retourne aucune
                ligne (ce qui indiquerait une forme de reponse Supabase
                inattendue).
        """
        # Valide et normalise l'entree via le modele Pydantic avant
        # d'envoyer quoi que ce soit a Supabase. Cela permet de detecter
        # les donnees invalides (par exemple une valeur de `source`
        # incorrecte) tot, avec une erreur claire, plutot que de laisser
        # Postgres la rejeter avec un message cryptique.
        memory = UserMemoryCreate(
            user_id=user_id,
            content=content,
            source=source,
            metadata=metadata or {},
        )
        payload = memory.model_dump(mode="json")

        try:
            response = self._client.table(USER_MEMORIES_TABLE).insert(payload).execute()
        except Exception as exc:  # noqa: BLE001 - Les exceptions du SDK Supabase varient selon la version.
            logger.exception("Creation memoire utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Creation memoire utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserMemoryServiceError("Creation memoire utilisateur impossible : aucune ligne retournee.")
        return self._to_memory_dict(row)

    def get_user_memories(self, user_id: UUID | str) -> list[dict[str, Any]]:
        """Retourne toutes les memoires d'un utilisateur.

        Args:
            user_id: L'utilisateur dont les memoires doivent etre listees.

        Returns:
            Une liste de dicts de memoires normalisees, dans l'ordre
            renvoye par Supabase (aucun ``order_by`` explicite n'est
            applique ici — ajoutez-en un si un ordre precis, par exemple
            par date de creation, est necessaire). Retourne une liste
            vide si l'utilisateur n'a encore aucune memoire ; ce n'est
            PAS une condition d'erreur.

        Raises:
            UserMemoryServiceError: Si l'appel de selection Supabase
                sous-jacent echoue (par exemple erreur reseau, table ou
                colonne manquante).
        """
        try:
            response = (
                self._client.table(USER_MEMORIES_TABLE)
                .select("*")
                .eq("user_id", str(user_id))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lecture memoires utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Lecture memoires utilisateur impossible : {exc}") from exc

        return [self._to_memory_dict(row) for row in self._rows(response)]

    def get_memory(self, user_id: UUID | str, memory_id: UUID | str) -> dict[str, Any]:
        """Retourne une memoire uniquement si elle appartient a l'utilisateur fourni.

        Args:
            user_id: Proprietaire attendu de la memoire. Sert de
                verification de controle d'acces, pas uniquement de cle
                de recherche.
            memory_id: Cle primaire de la ligne de memoire a recuperer.

        Returns:
            Le dict de memoire normalise.

        Raises:
            UserMemoryServiceError: Si l'appel de selection Supabase lui-
                meme echoue.
            UserMemoryNotFoundError: Si aucune ligne ne correspond a la
                fois a ``memory_id`` ET ``user_id`` — ce cas couvre a la
                fois "la memoire n'existe vraiment pas" et "la memoire
                existe mais appartient a quelqu'un d'autre".
        """
        try:
            response = (
                self._client.table(USER_MEMORIES_TABLE)
                .select("*")
                .eq("id", str(memory_id))
                .eq("user_id", str(user_id))
                .limit(1)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lecture memoire utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Lecture memoire utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserMemoryNotFoundError(f"Memoire utilisateur introuvable : {memory_id}")
        return self._to_memory_dict(row)

    def update_memory(
        self,
        user_id: UUID | str,
        memory_id: UUID | str,
        data: UserMemoryUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        """Met a jour une memoire uniquement dans le perimetre de l'utilisateur fourni.

        Prend en charge les mises a jour partielles : seuls les champs
        explicitement definis sur ``data`` sont envoyes a Supabase, ce
        qui permet par exemple de mettre a jour uniquement ``content``
        sans ecraser ``metadata`` avec ``None``.

        Args:
            user_id: Proprietaire attendu de la memoire ; utilise comme
                filtre de controle d'acces, exactement comme dans
                :meth:`get_memory`.
            memory_id: Cle primaire de la ligne de memoire a mettre a jour.
            data: Soit une instance de :class:`UserMemoryUpdate`, soit un
                dict brut de meme forme. Les champs non definis ou
                explicitement a ``None`` sont retires avant l'envoi de la
                mise a jour, grace a
                ``exclude_none=True, exclude_unset=True``.

        Returns:
            Le dict de memoire normalise reflechissant la ligne *apres*
            la mise a jour.

        Raises:
            ValueError: Si, apres avoir retire les champs non definis ou
                a ``None``, il ne reste plus rien a mettre a jour (un
                payload vide serait sinon un appel Supabase sans effet,
                ce qui est presque certainement un bug cote appelant a
                signaler immediatement).
            UserMemoryServiceError: Si l'appel de mise a jour Supabase
                lui-meme echoue.
            UserMemoryNotFoundError: Si aucune ligne ne correspond a la
                fois a ``memory_id`` ET ``user_id`` (meme raisonnement
                que dans :meth:`get_memory`).
        """
        # model_validate accepte soit un UserMemoryUpdate deja construit,
        # soit un simple dict, afin que les appelants n'aient pas a
        # construire le modele eux-memes pour les cas simples.
        update_model = UserMemoryUpdate.model_validate(data)
        payload = update_model.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        if not payload:
            raise ValueError("Aucune donnee de memoire utilisateur a mettre a jour.")

        try:
            response = (
                self._client.table(USER_MEMORIES_TABLE)
                .update(payload)
                .eq("id", str(memory_id))
                .eq("user_id", str(user_id))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Mise a jour memoire utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Mise a jour memoire utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserMemoryNotFoundError(f"Memoire utilisateur introuvable : {memory_id}")
        return self._to_memory_dict(row)

    def delete_memory(self, user_id: UUID | str, memory_id: UUID | str) -> None:
        """Supprime une memoire uniquement dans le perimetre de l'utilisateur fourni.

        Args:
            user_id: Proprietaire attendu de la memoire ; utilise comme
                filtre de controle d'acces, exactement comme dans
                :meth:`get_memory`.
            memory_id: Cle primaire de la ligne de memoire a supprimer.

        Returns:
            None. Le succes est signale par l'absence d'exception.

        Raises:
            UserMemoryServiceError: Si l'appel de suppression Supabase
                lui-meme echoue.
            UserMemoryNotFoundError: Si aucune ligne ne correspondait a la
                fois a ``memory_id`` ET ``user_id`` — la commande
                ``delete`` de Supabase retourne les lignes supprimees,
                donc un resultat vide signifie qu'aucune ligne
                correspondant au filtre n'existait reellement a
                supprimer.
        """
        try:
            response = (
                self._client.table(USER_MEMORIES_TABLE)
                .delete()
                .eq("id", str(memory_id))
                .eq("user_id", str(user_id))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Suppression memoire utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Suppression memoire utilisateur impossible : {exc}") from exc

        if self._first_row(response) is None:
            raise UserMemoryNotFoundError(f"Memoire utilisateur introuvable : {memory_id}")

    @staticmethod
    def _rows(response: Any) -> list[dict[str, Any]]:
        """Extrait les lignes d'un objet de reponse Supabase ou d'un mapping.

        Selon les versions, le SDK Python de Supabase a retourne le
        resultat de ``.execute()`` soit sous forme de simple ``dict``
        avec une cle ``"data"``, soit sous forme d'objet de type
        ``APIResponse`` exposant un attribut ``.data``. Cette fonction
        utilitaire normalise les deux formes en une liste plate de dicts
        de lignes, afin que le reste du service n'ait pas a se soucier de
        la version du SDK installee.

        Args:
            response: L'objet brut retourne par ``.execute()``.

        Returns:
            Une liste de dicts de lignes. Retourne une liste vide si
            ``data`` vaut ``None`` (par exemple une suppression qui n'a
            touche aucune ligne). Si ``data`` est un simple dict plutot
            qu'une liste, il est enveloppe dans une liste a un seul
            element par souci de coherence.

        Raises:
            UserMemoryServiceError: Si ``data`` est present mais n'est ni
                une liste ni un dict, ce qui indiquerait une forme de
                reponse du SDK Supabase inattendue, que ce code ne sait
                pas gerer.
        """
        data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise UserMemoryServiceError("Reponse Supabase inattendue pour la table user_memories.")

    @classmethod
    def _first_row(cls, response: Any) -> dict[str, Any] | None:
        """Retourne la premiere ligne d'une reponse Supabase, s'il y en a une.

        Fine fonction de confort autour de :meth:`_rows` pour le cas tres
        frequent (insertion / lecture unique / mise a jour / suppression)
        ou l'appelant ne s'interesse qu'a une seule ligne, ou simplement
        a savoir si une ligne quelconque a correspondu.

        Args:
            response: L'objet brut retourne par ``.execute()``.

        Returns:
            Le dict de la premiere ligne, ou ``None`` si la reponse ne
            contenait aucune ligne.
        """
        rows = cls._rows(response)
        return rows[0] if rows else None

    @staticmethod
    def _to_memory_dict(row: dict[str, Any]) -> dict[str, Any]:
        """Normalise une ligne Supabase en un dict de memoire pret pour le JSON.

        Fait transiter la ligne brute par le modele Pydantic
        :class:`UserMemory`. Cela permet a la fois de valider que
        Supabase a bien retourne des donnees conformes au schema attendu,
        et d'assurer une serialisation JSON coherente (par exemple les
        champs ``UUID`` et ``datetime`` rendus sous forme de chaines)
        quelle que soit la maniere dont le SDK Supabase les represente en
        interne.

        Args:
            row: Une seule ligne brute telle que retournee par le SDK
                Supabase.

        Returns:
            Un dict serialisable en JSON correspondant a la forme
            publique ``UserMemory``, pret a etre retourne directement
            dans une reponse d'API.
        """
        return UserMemory.model_validate(row).model_dump(mode="json")


# Singleton cree de maniere paresseuse, support des fonctions utilitaires
# de module ci-dessous. Conserve comme etat de module (plutot
# qu'instancie a l'import) afin que l'import de ce module ne declenche
# jamais de connexion au client Supabase — utile pour la collecte des
# tests et pour tout contexte ou les variables d'environnement requises
# par get_supabase_admin_client() ne seraient pas encore configurees.
_default_service: UserMemoryService | None = None


def _get_default_service() -> UserMemoryService:
    """Retourne une instance de service par defaut, initialisee paresseusement.

    Returns:
        Une instance de :class:`UserMemoryService` partagee au niveau du
        processus, creee au premier appel et reutilisee a chaque appel
        suivant.
    """
    global _default_service
    if _default_service is None:
        _default_service = UserMemoryService()
    return _default_service


def create_memory(
    user_id: UUID | str,
    content: str,
    source: UserMemorySource | str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cree une memoire pour un utilisateur en utilisant le service par defaut.

    Fonction utilitaire encapsulant ``UserMemoryService.create_memory``
    pour les appelants qui n'ont pas besoin de gerer eux-memes une
    instance du service (par exemple les handlers de routes FastAPI).
    Voir cette methode pour la documentation complete des parametres et
    exceptions.
    """
    return _get_default_service().create_memory(user_id, content, source, metadata)


def get_user_memories(user_id: UUID | str) -> list[dict[str, Any]]:
    """Retourne toutes les memoires d'un utilisateur en utilisant le service par defaut.

    Fonction utilitaire encapsulant ``UserMemoryService.get_user_memories``.
    Voir cette methode pour la documentation complete des parametres et
    exceptions.
    """
    return _get_default_service().get_user_memories(user_id)


def get_memory(user_id: UUID | str, memory_id: UUID | str) -> dict[str, Any]:
    """Retourne une memoire d'un utilisateur en utilisant le service par defaut.

    Fonction utilitaire encapsulant ``UserMemoryService.get_memory``. Voir
    cette methode pour la documentation complete des parametres et
    exceptions.
    """
    return _get_default_service().get_memory(user_id, memory_id)


def update_memory(user_id: UUID | str, memory_id: UUID | str, data: UserMemoryUpdate | dict[str, Any]) -> dict[str, Any]:
    """Met a jour une memoire d'un utilisateur en utilisant le service par defaut.

    Fonction utilitaire encapsulant ``UserMemoryService.update_memory``.
    Voir cette methode pour la documentation complete des parametres et
    exceptions.
    """
    return _get_default_service().update_memory(user_id, memory_id, data)


def delete_memory(user_id: UUID | str, memory_id: UUID | str) -> None:
    """Supprime une memoire d'un utilisateur en utilisant le service par defaut.

    Fonction utilitaire encapsulant ``UserMemoryService.delete_memory``.
    Voir cette methode pour la documentation complete des parametres et
    exceptions.
    """
    _get_default_service().delete_memory(user_id, memory_id)